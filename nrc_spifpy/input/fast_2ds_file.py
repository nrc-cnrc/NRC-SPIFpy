
import os
import numpy
import pandas as pd
from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.images import Images

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time
import gc


try:
    from .fast_2ds_decoder import decode_frame
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

if os.environ.get('FORCE_PYTHON_F2DS') == '1':
    HAS_CYTHON = False

class Fast2DSFile(BinaryFile):
    """
    Class for reading and processing SPEC Fast 2D-S (Type 48) probe data.

    Reference: SPEC_OAP_Data_File_Formats_July_2022_Rev_D

    File Structure:
        - Base file (.F2DS): Contains image packets (2S) and NULL packets (NL)
        - HK file (.F2DSHK): Contains housekeeping (HK) and mask (MK) packets

    Data Block Format (4114 bytes):
        - Timestamp: 16 bytes (8 x uint16: year, month, dow, day, hour, min, sec, ms)
        - Raw Data: 4096 bytes (2048 x uint16 words containing packets)
        - Checksum: 2 bytes (discarded)

    Image Packet Format:
        - Word 1: Header "2S" (0x3253 = 12883 decimal)
        - Word 2: NHraw (bits 11-0: word count, bit 12: multi-packet, bit 15: overflow)
        - Word 3: NVraw (same format as NHraw)
        - Word 4: Particle ID (16-bit counter)
        - Word 5: Slice count (number of 128-pixel slices)
        - Words 6 to 5+N-3: Compressed/uncompressed image data
        - Words 5+N-2 to 5+N: 48-bit timing (only if bit 12 = 0)

    Image Data Encoding:
        - 0x4000: Fully shaded slice (128 shaded pixels)
        - 0x7FFF: Next 8 words are uncompressed (8 x 16 bits = 128 pixels)
        - Otherwise: RLE word (bit 14: start slice, bits 13-7: shaded, bits 6-0: clear)

    Attributes:
        diodes (int): Number of diodes (128 for 2D-S)
        aux_channels (list): Auxiliary data channels from HK file

    History:
        - 2026-07-15: Yongjie Huang, implemented probe-counter timing after discussion
                      with Aaron Bansemer (NSF NCAR) and Parker Morris (SPEC).
        - 2026-01-11: Yongjie Huang, first implementation.
    """

    # -------------------------------------------------------------------------
    # Format Constants
    # -------------------------------------------------------------------------
    SYNC_2S = 12883           # 0x3253: Image packet header
    SYNC_NL = 0x4E4C          # NULL packet header (disk bytes 4C 4E)
    RLE_FULL_SHADED = 0x4000  # Fully shaded slice (128 pixels)
    RLE_UNCOMPRESSED = 0x7FFF # Next 8 words are raw bitmap
    TIMING_MODULUS = 1 << 48  # 2**48
    TIMING_OFFSET_WINDOW_SECONDS = 300.0
    MAX_ANCHOR_DISTANCE_SECONDS = 10.0
    TIMING_HK_COUNTER = 1
    TIMING_BUFFER_COUNTER = 2
    TIMING_BUFFER_ONLY = 3
    TIMING_INVALID_BUFFER = 4

    AUX_DTYPES = {
        'clock_counts': 'u8',
        'tas': 'f4',
        'user_temp': 'f4',
        'ps_temp': 'f4',
        'timing_quality': 'u1',
        'overload_flag': 'u1',
    }
    AUX_ATTRS = {
        'clock_counts': {
            'long_name': '48-bit probe count at the last image slice',
            'units': '1',
            'comment': 'Free-running counter stored modulo 2^48.',
        },
        'tas': {
            'long_name': (
                'True airspeed interpolated from external housekeeping records'
            ),
            'units': 'm/s',
            'comment': (
                'Housekeeping TAS interpolated to image-buffer timestamps and '
                'copied to each particle.'
            ),
        },
        'user_temp': {
            'long_name': (
                'DSP board temperature interpolated from external '
                'housekeeping records'
            ),
            'units': 'degree_Celsius',
        },
        'ps_temp': {
            'long_name': (
                'Power supply temperature interpolated from external '
                'housekeeping records'
            ),
            'units': 'degree_Celsius',
        },
        'timing_quality': {
            'long_name': 'particle timing source and quality',
            'flag_values': numpy.array([
                TIMING_HK_COUNTER,
                TIMING_BUFFER_COUNTER,
                TIMING_BUFFER_ONLY,
                TIMING_INVALID_BUFFER,
            ], dtype=numpy.uint8),
            'flag_meanings': (
                'hk_anchored_probe_counter '
                'buffer_anchored_probe_counter '
                'buffer_timestamp '
                'invalid_buffer_index'
            ),
        },
        'overload_flag': {
            'long_name': 'Probe particle overload status',
            'units': '1',
            'flag_values': numpy.array([0, 1], dtype=numpy.uint8),
            'flag_meanings': 'not_overloaded overloaded',
            'comment': (
                'Active-high flag decoded from bit 15 of the channel header.'
            ),
        },
    }

    # =========================================================================
    # Phase 1: Initialization
    # =========================================================================

    def __init__(
        self,
        filename,
        inst_name,
        resolution,
        clock_resolution_um=10.0,
    ):
        super().__init__(filename, inst_name, resolution)
        self.diodes = 128  # 2D-S has 128 photodiodes
        self.clock_resolution_um = float(clock_resolution_um)
        self.hk_counts = numpy.array([], dtype=numpy.uint64)
        self.hk_datetimes = numpy.array([], dtype='datetime64[ns]')
        self.hk_tas = numpy.array([], dtype=numpy.float64)

        # Data block dtype: 4114 bytes = 16 (timestamp) + 4096 (data) + 2 (checksum)
        self.file_dtype = numpy.dtype([
            ('year', 'u2'),      # Timestamp word 1
            ('month', 'u2'),     # Timestamp word 2
            ('weekday', 'u2'),   # Timestamp word 3 (0=Sun, 6=Sat)
            ('day', 'u2'),       # Timestamp word 4
            ('hour', 'u2'),      # Timestamp word 5
            ('minute', 'u2'),    # Timestamp word 6
            ('second', 'u2'),    # Timestamp word 7
            ('ms', 'u2'),        # Timestamp word 8 (milliseconds)
            ('data', '(2048,)u2'),  # Raw data frame (4096 bytes)
            ('discard', 'u2')    # Checksum (not used)
        ])

        # Auxiliary channels interpolated from external HK file
        self.aux_channels = [
            'clock_counts',
            'tas',
            'user_temp',
            'ps_temp',
            'timing_quality',
            'overload_flag',
        ]

    # =========================================================================
    # Phase 2: File Reading
    # =========================================================================

    def read(self):
        """ Read the base file and the separate HK file. """
        super().read() # Parent reads the base file into self.data

        # Child reads external HK file
        # For .F2DS files, HK is .F2DSHK (filename + 'HK')
        hk_filename = str(self.filename) + 'HK'

        if os.path.exists(hk_filename):
            self.hk_data = self._read_external_hk(hk_filename)
        else:
            print(f"Warning: HK file {hk_filename} not found.")
            self.hk_data = None

        if self.hk_data is not None:
             self._align_hk_to_frames() # Child aligns TAS to image frames
        else:
             print("Warning: Initializing empty auxiliary data (HK missing/failed).")
             n_frames = len(self.datetimes)
             self.tas = numpy.full(n_frames, numpy.nan)
             self.user_temp = numpy.full(n_frames, numpy.nan)
             self.ps_temp = numpy.full(n_frames, numpy.nan)

    def _read_external_hk(self, filename):
        """
        Reads the Type 48 Fast 2DS Housekeeping file.
        File structure: 72-byte Mask Pack + N x 182-byte HouseKeeping Packet records.
        Each 72-byte Mask Pack record: Timestamp (16 bytes) + Raw (54 bytes) + Checksum (2 bytes)
        Each 182-byte HouseKeeping Packet record: Timestamp (16 bytes) + Raw (164 bytes) + Checksum (2 bytes).
        """
        hk_dtype = numpy.dtype([('ts_year', '<u2'), ('ts_month', '<u2'), ('ts_dow', '<u2'), ('ts_day', '<u2'),
                                ('ts_hour', '<u2'), ('ts_min', '<u2'), ('ts_sec', '<u2'), ('ts_ms', '<u2'),
                                ('data', '(82,)<u2'),  # 164 bytes = 82 words
                                ('checksum', '<u2')])

        try:
            # Detect header size based on file size
            file_size = os.path.getsize(filename)
            remainder = file_size % 182

            offset = 0
            if remainder == 72:
                offset = 72
            elif remainder == 0:
                offset = 0
            else:
                print(f"Warning: HK file size {file_size} has remainder {remainder} (not 0 or 72). Assuming 72 and truncating.")
                offset = 72

            # Skip header, then read 182-byte records
            with open(filename, 'rb') as f:
                if offset > 0:
                    f.seek(offset)
                raw = f.read()

            # Truncate to multiple of 182 bytes
            read_len = len(raw)
            trunc_rem = read_len % 182
            if trunc_rem != 0:
                 print(f"Warning: HK read content (size {read_len}) not aligned to 182 bytes. Truncating {trunc_rem} bytes.")
                 raw = raw[:-trunc_rem]

            raw_hk = numpy.frombuffer(raw, dtype=hk_dtype)
            return raw_hk
        except Exception as e:
            print(f"Error reading HK file: {e}")
            return None


    def calc_buffer_datetimes(self):
        """ Calculates datetimes from buffers read in from file.
        Override to include milliseconds using vectorized pandas/numpy.
        """
        # Ensure start_date is set
        if not hasattr(self, 'start_date') or self.start_date is None:
             self.get_start_date()

        # Build datetime64 array directly from buffer fields
        df = pd.DataFrame({
            'year': self.data['year'],
            'month': self.data['month'],
            'day': self.data['day'],
            'hour': self.data['hour'],
            'minute': self.data['minute'],
            'second': self.data['second'],
            'microsecond': self.data['ms'].astype(numpy.int64) * 1000  # ms -> us
        })
        self.datetimes = pd.to_datetime(df).values  # numpy datetime64[ns] array


    def _align_hk_to_frames(self):
        """
        Interpolates HK data (TAS, Temps) to the timestamps of the image frames.
        """
        if self.hk_data is None or self.datetimes is None:
            return

        # 1. Convert HK timestamps to unix timestamps (Vectorized)
        data = self.hk_data

        # Extract components safely
        years = data['ts_year'].astype(numpy.int32)
        months = data['ts_month'].astype(numpy.int32)
        days = data['ts_day'].astype(numpy.int32)
        hours = data['ts_hour'].astype(numpy.int32)
        minutes = data['ts_min'].astype(numpy.int32)
        seconds = data['ts_sec'].astype(numpy.int32)
        ms = data['ts_ms'].astype(numpy.int32)

        # Validity mask to filter bad records
        valid_mask = (
            (months >= 1) & (months <= 12) &
            (days >= 1) & (days <= 31) &
            (hours >= 0) & (hours <= 23) &
            (minutes >= 0) & (minutes <= 59) &
            (seconds >= 0) & (seconds <= 59) &
            (years >= 2000) & (years <= 2100)
        )

        valid_indices = numpy.nonzero(valid_mask)[0]

        if len(valid_indices) == 0:
            print("Warning: No valid HK timestamps found.")
            self.tas = numpy.full(len(self.datetimes), numpy.nan)
            self.user_temp = numpy.full(len(self.datetimes), numpy.nan)
            self.ps_temp = numpy.full(len(self.datetimes), numpy.nan)
            return

        # Convert to unix timestamps
        ts_df = pd.DataFrame({
            'year': years[valid_indices],
            'month': months[valid_indices],
            'day': days[valid_indices],
            'hour': hours[valid_indices],
            'minute': minutes[valid_indices],
            'second': seconds[valid_indices],
            'microsecond': ms[valid_indices] * 1000  # ms -> us
        })

        self.hk_datetimes = pd.to_datetime(ts_df).values.astype('datetime64[ns]')
        hk_ts = self.hk_datetimes.astype(numpy.float64)

        if len(hk_ts) == 0:
            print("Warning: No valid HK timestamps found after conversion.")
            self.tas = numpy.full(len(self.datetimes), numpy.nan)
            self.user_temp = numpy.full(len(self.datetimes), numpy.nan)
            self.ps_temp = numpy.full(len(self.datetimes), numpy.nan)
            return

        # 2. Convert Frame timestamps to nanoseconds (float64)
        frame_ts = self.datetimes.astype(numpy.float64)

        # 3. Extract and Interpolate Channels (use only valid HK records)

        # --- TAS (True Air Speed) ---
        # TAS is at words 75,76 in the 82-word data section (empirically verified)
        # Word 75 is MSW (17194), Word 76 is LSW (0) -> (w75 << 16) | w76 = 170.00 m/s

        hk_words = self.hk_data['data'][valid_indices]
        self.hk_counts = (
            (hk_words[:, 72].astype(numpy.uint64) << 32)
            | (hk_words[:, 73].astype(numpy.uint64) << 16)
            | hk_words[:, 74].astype(numpy.uint64)
        )

        tas_w75 = hk_words[:, 75]
        tas_w76 = hk_words[:, 76]

        # Combine to 32-bit int
        tas_int = (tas_w75.astype(numpy.uint32) << 16) | tas_w76.astype(numpy.uint32)
        hk_tas = tas_int.view(numpy.float32).astype(numpy.float64)
        self.hk_tas = hk_tas

        # Filter out invalid TAS values and interpolate
        valid_tas_mask = (hk_tas > 0) & (hk_tas < 500)

        if numpy.any(valid_tas_mask):
            self.tas = numpy.interp(frame_ts, hk_ts[valid_tas_mask], hk_tas[valid_tas_mask])
        else:
            print("Warning: No valid TAS values found in HK data.")
            self.tas = numpy.full(len(self.datetimes), numpy.nan)

        # --- Temperatures ---
        # DSP Board Temp (Word 17) as 'user_temp' and Power Supply (Word 22) as 'ps_temp'
        # Conversion: C0=-64.8, C1=0.07323

        raw_dsp = self.hk_data['data'][valid_indices, 16]
        dsp_temp = raw_dsp * 0.07323 - 64.8
        self.user_temp = numpy.interp(frame_ts, hk_ts, dsp_temp)

        # Power Supply Word 22 (Index 21)
        raw_ps = self.hk_data['data'][valid_indices, 21]
        ps_temp = raw_ps * 0.07323 - 64.8
        self.ps_temp = numpy.interp(frame_ts, hk_ts, ps_temp)


    # =========================================================================
    # Phase 3: Parallel Processing (Main Entry Point)
    # =========================================================================

    def _write_images(self, spiffile, inst_name, images):
        """Write Fast 2D-S images while preserving the 48-bit clock count."""
        writer = getattr(spiffile, 'write_images_with_extra_aux_dtypes', None)
        if writer is None:
            spiffile.write_images(inst_name, images)
        else:
            writer(inst_name, images, self.AUX_DTYPES, self.AUX_ATTRS)

    def process_file(self, spiffile, processors=None):
        """
        Orchestrates parallel processing of the file.
        Uses per-chunk state for cross-frame particle handling within each parallel worker.

        History:
            - 2026-01-11: Yongjie Huang, first implementation.
        """
        if HAS_CYTHON:
            print("Using cythonized decoder")
        else:
            print("Using pure python decoder")

        if processors is None:
            processors = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

        spiffile.set_start_date(self.start_date.strftime('%Y-%m-%d %H:%M:%S %z'))

        # Create Output Groups
        spiffile.create_inst_group(self.name + '-H')
        spiffile.create_inst_group(self.name + '-V')
        spiffile.set_filenames_attr(self.name + '-H', self.filename)
        spiffile.set_filenames_attr(self.name + '-V', self.filename)

        # Write buffer info
        spiffile.write_buffer_info(self.start_date, self.datetimes)

        # Setup parallel processing
        process_until = len(self.data)
        chunksize = 500
        data_chunk = range(0, process_until)

        pbar1 = tqdm(desc='Processing frames', total=process_until, unit=' frame')
        pbar2 = tqdm(desc='Writing frames', total=process_until, unit=' frame')

        futures = []
        max_write_queue = 8
        images_remaining = process_until > 0
        i = 0

        tot_h = 0
        tot_v = 0
        t00 = time.time()

        with ProcessPoolExecutor(max_workers=processors) as executor:
            while True:
                while len(futures) <= max_write_queue and images_remaining:
                    chunk = data_chunk[i: i + chunksize]
                    futures.append(executor.submit(self.process_frames, chunk))
                    i += chunksize
                    if i >= process_until:
                        images_remaining = False
                    pbar1.update(len(chunk))

                if futures:
                    # Take the oldest submitted chunk to preserve frame order.
                    future = futures.pop(0)
                    # Wait for that worker and unpack its H/V image results.
                    h_imgs, v_imgs = future.result()
                    pbar2.update(min(chunksize, process_until - pbar2.n))

                    tot_h += len(h_imgs)
                    tot_v += len(v_imgs)

                    # Write Images (must call conv_to_array first to flatten)
                    if len(h_imgs) > 0:
                        h_imgs.conv_to_array(self.diodes)
                        self._write_images(spiffile, self.name + '-H', h_imgs)
                    if len(v_imgs) > 0:
                        v_imgs.conv_to_array(self.diodes)
                        self._write_images(spiffile, self.name + '-V', v_imgs)

                    gc.collect()

                if not images_remaining and len(futures) == 0:
                    break

        pbar1.close()
        pbar2.close()

        if hasattr(spiffile, 'instgrps'):
            self._finalize_particle_times(spiffile, self.name + '-H')
            self._finalize_particle_times(spiffile, self.name + '-V')

        print(f'Finished. {tot_h}-H, {tot_v}-V images processed in {time.time()-t00:.2f}s')


    def process_frames(self, chunk):
        """
        Process a chunk of frames indices.
        State is initialized per-chunk for parallel processing compatibility.

        History:
            - 2026-01-11: Yongjie Huang, first implementation.
        """
        h_accum = []
        v_accum = []

        # Initialize per-chunk state (not shared with other chunks)
        # We rely on next_start_idx to handle frame boundaries logic locally
        chunk_state = {
            'pending_h': {},
            'pending_v': {},
        }

        next_start_idx = 0
        for frame_idx in chunk:
            result = self.process_frame_with_state(frame_idx, chunk_state, start_idx=next_start_idx)
            h_accum.extend(result['h'])
            v_accum.extend(result['v'])
            next_start_idx = result['next_idx']

        buffer = {'h': h_accum, 'v': v_accum}
        return self.extract_images(buffer)

    # =========================================================================
    # Phase 4: Frame Decoding
    # =========================================================================

    def process_frame_with_state(self, frame, chunk_state, start_idx=0):
        """
        Decodes a single 4096-byte frame (stored as 2048 'u2' words).
        Handles multi-packet particles and cross-frame spanning using Look-Ahead.

        Parameters
        ----------
        frame : int
            Index of the current frame in self.data.
        chunk_state : dict
            Dictionary holding pending multi-packet particle states.
        start_idx : int, optional
            Index to start processing from within the frame (skipping data consumed by previous frame).

        Returns
        -------
        dict
            {'h': h_images, 'v': v_images, 'next_idx': next_start_idx}

        History:
            - 2026-01-11: Yongjie Huang, first implementation.
        """
        record = self.data[frame]['data']

        # Look ahead availability
        try:
            record_next = self.data[frame+1]['data']
            next_record_exists = True
        except (IndexError, KeyError):
            record_next = numpy.empty(0, dtype=numpy.uint16)
            next_record_exists = False

        if HAS_CYTHON:
             # Delegate to Cython implementation
             return decode_frame(record, record_next, next_record_exists, chunk_state, frame, start_idx)

        h_images = []
        v_images = []

        idx = start_idx
        limit = len(record) # Limit for processing loop
        reach_record_end = False
        next_start_idx = 0

        # Main particle scanning loop
        while idx < limit:
            word = record[idx]

            # Check for '2S' image packet sync word (0x3253 = 12883)
            if word == self.SYNC_2S:
                # Parse 5-word header: [2S, NHraw, NVraw, PID, Slices]
                if idx + 5 <= limit:
                    nh_temp = record[idx+1] & 0x0FFF
                    nv_temp = record[idx+2] & 0x0FFF
                    n_temp = nh_temp if nh_temp > 0 else nv_temp

                    if idx + 5 + n_temp > limit:
                        reach_record_end = True
                else:
                    reach_record_end = True

                if reach_record_end:
                    if next_record_exists:
                        # Extend record with next frame
                        record = numpy.concatenate((record, record_next))
                        # Note: 'limit' still refers to original frame boundary
                    else:
                        break  # EOF

                # Decode Header
                nh_raw = record[idx+1]
                nv_raw = record[idx+2]
                pid = record[idx+3]
                slices = record[idx+4]

                # Extract word count from bits 11-0 (mask 0x0FFF = 4095)
                nh = nh_raw & 0x0FFF
                nv = nv_raw & 0x0FFF
                n_words = nh if nh > 0 else nv  # One of NH/NV is always 0

                if n_words == 0:
                    idx += 1
                    continue

                is_horiz = (nh > 0)
                # Bit 12 (0x1000): multi-packet flag (1 = more packets follow, no timing words)
                channel_header = nh_raw if is_horiz else nv_raw
                is_multi_packet = (channel_header & 0x1000) >> 12
                overload_flag = (channel_header & 0x8000) >> 15

                data_start = idx + 5
                data_end = data_start + n_words

                # A packet may use at most the current and one look-ahead frame.
                # Reject truncated or malformed lengths instead of decoding a
                # partial payload or carrying an invalid offset forward.
                if data_end > len(record):
                    print(
                        f'Warning: Fast 2DS packet at frame {frame}, word {idx} '
                        f'spans more than two frames (n_words={n_words}); '
                        'packet skipped.'
                    )
                    break

                # Packet data
                full_packet_data = record[data_start:data_end]

                # --- Decoding Logic ---

                pending = chunk_state['pending_h'] if is_horiz else chunk_state['pending_v']
                if pid not in pending:
                    pending[pid] = {'img_decomp': [], 'slice_decomp': [], 'non_compressed': 0}
                state = pending[pid]

                # Process Data
                # Per spec: if bit 12 set (multi-packet), NO timing words at end
                # Otherwise: last 3 words are 48-bit timing
                if is_multi_packet:
                    # Multi-packet: all N words are image data, no timing
                    timing = 0
                    payload_data = full_packet_data
                elif len(full_packet_data) >= 3:
                    # Single/final packet: extract 48-bit timing from last 3 words
                    timing = ((int(full_packet_data[-1]) << 32) |
                              (int(full_packet_data[-2]) << 16) |
                              (int(full_packet_data[-3])))
                    payload_data = full_packet_data[:-3]
                else:
                    timing = 0  # Shouldn't happen but handle gracefully
                    payload_data = full_packet_data

                # Decode image data (RLE compressed or raw bitmap)
                for val in payload_data:
                    val = int(val)

                    if state['non_compressed'] > 0:
                        # Raw bitmap mode: convert 16-bit word to 16 pixels
                        # Pixel format: 1=clear, 0=shaded (inverted at final output)
                        bin_line = [-1 * (int(n) - 1) for n in bin(val)[2:].zfill(16)[::-1]]
                        state['slice_decomp'].extend(bin_line)
                        state['non_compressed'] -= 1
                        if state['non_compressed'] == 0 and len(state['slice_decomp']) > 0:
                            # Pad slice to 128 pixels and finalize
                            if len(state['slice_decomp']) % 128 > 0:
                                state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                            state['img_decomp'].extend(state['slice_decomp'])
                            state['slice_decomp'] = []

                    elif val == self.RLE_UNCOMPRESSED:  # 0x7FFF
                        # Start of uncompressed slice: next 8 words are raw 128-bit bitmap
                        if len(state['slice_decomp']) > 0:
                            if len(state['slice_decomp']) % 128 > 0:
                                state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                            state['img_decomp'].extend(state['slice_decomp'])
                            state['slice_decomp'] = []
                        state['non_compressed'] = 8  # Next 8 words are raw bitmap

                    elif val == self.RLE_FULL_SHADED:  # 0x4000
                        # Fully shaded slice: 128 shaded pixels
                        if len(state['slice_decomp']) > 0:
                            if len(state['slice_decomp']) % 128 > 0:
                                state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                            state['img_decomp'].extend(state['slice_decomp'])
                            state['slice_decomp'] = []
                        state['img_decomp'].extend([1] * 128)  # All shaded

                    else:
                        # RLE encoded word:
                        #   bit 15: always 0 for RLE
                        #   bit 14: start of new slice (1 = first word of slice)
                        #   bits 13-7: number of shaded pixels (0-127)
                        #   bits 6-0: number of clear pixels (0-127)
                        startslice = (val >> 14) & 1
                        num_shaded = (val >> 7) & 0x7F   # bits 13-7
                        num_clear = val & 0x7F           # bits 6-0

                        if startslice == 1 and len(state['slice_decomp']) > 0:
                            # Start of new slice - finalize previous
                                if len(state['slice_decomp']) % 128 > 0:
                                    state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                                state['img_decomp'].extend(state['slice_decomp'])
                                state['slice_decomp'] = []
                        # Append clear pixels then shaded pixels
                        state['slice_decomp'].extend([0] * num_clear)
                        state['slice_decomp'].extend([1] * num_shaded)


                # Finalize Image
                if not is_multi_packet:
                    if len(state['slice_decomp']) > 0:
                        if len(state['slice_decomp']) % 128 > 0:
                            state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                        state['img_decomp'].extend(state['slice_decomp'])

                    final_buffer = bytearray(state['img_decomp'])
                    final_data = numpy.frombuffer(final_buffer, dtype=numpy.uint8)
                    numpy.bitwise_xor(final_data, 1, out=final_data)

                    img_result = {
                        'id': pid,
                        'slices': slices,
                        'time': timing,
                        'data': final_data,
                        'buffer_index': frame,
                        'overload_flag': overload_flag,
                    }
                    if is_horiz:
                        h_images.append(img_result)
                    else:
                        v_images.append(img_result)
                    del pending[pid]

                # Exit loop after spanning particle (it was the last one in this frame)
                if reach_record_end:
                    next_start_idx = data_end - limit
                    break

                idx = data_end

            elif word == self.SYNC_NL:  # 0x4E4C: NULL packet (FIFO flush/padding)
                idx += 1
            else:
                # Unknown word - skip
                idx += 1

        return {'h': h_images, 'v': v_images, 'next_idx': next_start_idx}


    # =========================================================================
    # Phase 5: Timing and Image Conversion
    # =========================================================================

    def _seconds_from_start(self, datetimes):
        """Convert datetime64 values to floating-point seconds from file start."""
        start = pd.Timestamp(self.start_date)
        if start.tzinfo is not None:
            start = start.tz_convert('UTC').tz_localize(None)
        start64 = start.to_datetime64().astype('datetime64[ns]')
        values = numpy.asarray(datetimes, dtype='datetime64[ns]')
        return (values - start64) / numpy.timedelta64(1, 's')

    @classmethod
    def _signed_counter_delta(cls, current, previous):
        """Return the shortest signed difference between 48-bit counters."""
        current = numpy.asarray(current, dtype=numpy.int64)
        previous = numpy.asarray(previous, dtype=numpy.int64)
        delta = current - previous
        half = cls.TIMING_MODULUS // 2
        return (delta + half) % cls.TIMING_MODULUS - half

    def _unwrap_counter(self, counts):
        """Convert one monotonic 48-bit counter segment to increasing integers."""
        raw_counts = numpy.asarray(counts, dtype=numpy.int64)
        count_delta = self._signed_counter_delta(
            raw_counts[1:], raw_counts[:-1]
        )
        if numpy.any(count_delta <= 0):
            return None, None

        unwrapped = numpy.empty(len(raw_counts), dtype=numpy.int64)
        unwrapped[0] = raw_counts[0]
        unwrapped[1:] = unwrapped[0] + numpy.cumsum(count_delta)
        return unwrapped, count_delta

    def _counter_elapsed_seconds(self, count_delta, tas):
        """Integrate probe-count increments into elapsed seconds using TAS.

        One count represents ``clock_resolution_um`` of flight distance. The
        mean endpoint TAS approximates the speed over each sampling interval.
        """
        tas = numpy.asarray(tas, dtype=numpy.float64)
        mean_tas = 0.5 * (tas[:-1] + tas[1:])
        # Δt = Δcount × y_resolution / TAS
        interval_seconds = (
            numpy.asarray(count_delta, dtype=numpy.float64)
            * self.clock_resolution_um
            * 1.0e-6
            / mean_tas
        )
        return numpy.concatenate(([0.0], numpy.cumsum(interval_seconds)))

    @staticmethod
    def _nearest_indices(sorted_values, values):
        """Return indexes of the nearest entries in a sorted 1-D array."""
        right = numpy.searchsorted(sorted_values, values, side='left')
        right = numpy.clip(right, 0, len(sorted_values) - 1)
        left = numpy.clip(right - 1, 0, len(sorted_values) - 1)
        choose_left = (
            numpy.abs(values - sorted_values[left])
            <= numpy.abs(sorted_values[right] - values)
        )
        return numpy.where(choose_left, left, right)

    @staticmethod
    def _seconds_to_sec_ns(seconds):
        """Split floating-point relative seconds into normalized sec/ns arrays."""
        seconds = numpy.asarray(seconds, dtype=numpy.float64)
        safe_seconds = numpy.where(numpy.isfinite(seconds), seconds, 0.0)
        sec = numpy.floor(safe_seconds).astype(numpy.int64)
        ns = numpy.rint((safe_seconds - sec) * 1.0e9).astype(numpy.int64)

        carry = ns >= 1_000_000_000
        sec[carry] += 1
        ns[carry] -= 1_000_000_000
        return sec, ns

    @staticmethod
    def _replace_backwards_times(
        particle_seconds,
        timing_quality,
        fallback_seconds,
        fallback_quality,
    ):
        """Replace backwards timestamp pairs until adjacent boundaries are valid.

        Replacing a pair can expose a new backwards jump between a fallback
        value and its unchanged neighbor. Recheck after each pair replacement
        until the timeline is nondecreasing or the fixed fallback cannot make
        further progress.
        """
        pending = numpy.nonzero(
            numpy.diff(particle_seconds) < -1.0e-6
        )[0].tolist()
        while pending:
            index = pending.pop()
            if (
                particle_seconds[index + 1] - particle_seconds[index]
                >= -1.0e-6
            ):
                continue

            before = particle_seconds[index:index + 2].copy()
            particle_seconds[index:index + 2] = fallback_seconds[index:index + 2]
            timing_quality[index:index + 2] = fallback_quality[index:index + 2]
            if numpy.array_equal(before, particle_seconds[index:index + 2]):
                continue

            # Only the neighboring edges can have become invalid. Checking
            # those locally avoids repeatedly scanning multi-million particles.
            for neighbor in (index - 1, index + 1):
                if (
                    0 <= neighbor < len(particle_seconds) - 1
                    and particle_seconds[neighbor + 1]
                    - particle_seconds[neighbor] < -1.0e-6
                ):
                    pending.append(neighbor)

    def _counter_segments(self, counts, seconds):
        """Split counter records at resets or backwards acquisition time jumps."""
        if len(counts) == 0:
            return []

        count_delta = self._signed_counter_delta(counts[1:], counts[:-1])
        time_delta = numpy.diff(seconds)
        breaks = numpy.nonzero((count_delta <= 0) | (time_delta < 0))[0] + 1
        edges = numpy.concatenate(([0], breaks, [len(counts)]))
        return [
            slice(int(start), int(stop))
            for start, stop in zip(edges[:-1], edges[1:])
            if stop - start >= 2
        ]

    def _smoothed_timing_offset(self, elapsed, acquisition_seconds):
        """Align precise relative counter time to noisy acquisition UTC time.

        Median offsets in fixed windows retain counter-based interarrival
        timing without copying short-period PC timestamp jitter.
        """
        residual = acquisition_seconds - elapsed
        window = self.TIMING_OFFSET_WINDOW_SECONDS
        window_id = numpy.floor(
            (acquisition_seconds - acquisition_seconds[0]) / window
        ).astype(numpy.int64)

        centers = []
        offsets = []
        for value in numpy.unique(window_id):
            selected = window_id == value
            centers.append(numpy.median(elapsed[selected]))
            offsets.append(numpy.median(residual[selected]))

        return (
            numpy.asarray(centers, dtype=numpy.float64),
            numpy.asarray(offsets, dtype=numpy.float64),
        )

    @staticmethod
    def _times_from_buffer_anchors(
        unwrapped_counts,
        buffer_indices,
        buffer_seconds,
    ):
        """Estimate count-to-time conversion from consecutive buffer anchors."""
        unique_buffers = numpy.unique(buffer_indices)
        if len(unique_buffers) < 2:
            return None

        # A median particle count is a stable representative for each buffer.
        anchor_counts = numpy.array([
            numpy.median(unwrapped_counts[buffer_indices == buffer_index])
            for buffer_index in unique_buffers
        ], dtype=numpy.float64)
        anchor_seconds = numpy.array([
            numpy.median(buffer_seconds[buffer_indices == buffer_index])
            for buffer_index in unique_buffers
        ], dtype=numpy.float64)

        increasing = numpy.concatenate((
            [True],
            (numpy.diff(anchor_counts) > 0)
            & (numpy.diff(anchor_seconds) > 0),
        ))
        anchor_counts = anchor_counts[increasing]
        anchor_seconds = anchor_seconds[increasing]
        if len(anchor_counts) < 2:
            return None

        count_values = unwrapped_counts.astype(numpy.float64)
        calculated = numpy.interp(
            count_values, anchor_counts, anchor_seconds
        )

        # numpy.interp clamps outside the anchors; use the endpoint slopes.
        left = count_values < anchor_counts[0]
        right = count_values > anchor_counts[-1]
        left_slope = (
            (anchor_seconds[1] - anchor_seconds[0])
            / (anchor_counts[1] - anchor_counts[0])
        )
        right_slope = (
            (anchor_seconds[-1] - anchor_seconds[-2])
            / (anchor_counts[-1] - anchor_counts[-2])
        )
        calculated[left] = (
            anchor_seconds[0]
            + (count_values[left] - anchor_counts[0]) * left_slope
        )
        calculated[right] = (
            anchor_seconds[-1]
            + (count_values[right] - anchor_counts[-1]) * right_slope
        )
        return calculated

    def _buffer_counter_fallback(
        self,
        timings,
        buffer_indices,
        buffer_seconds,
        valid_buffer,
    ):
        """Use buffer timestamps to anchor probe-counter interarrival times.

        TAS provides the preferred count-to-time conversion. If TAS is not
        available, estimate that conversion from consecutive buffer anchors.
        """
        particle_seconds = buffer_seconds.copy()
        timing_quality = numpy.full(
            len(timings), self.TIMING_INVALID_BUFFER, dtype=numpy.uint8
        )
        timing_quality[valid_buffer] = self.TIMING_BUFFER_ONLY
        raw_quality = timing_quality.copy()
        valid_index = numpy.nonzero(valid_buffer)[0]
        if len(valid_index) < 2:
            return particle_seconds, timing_quality

        valid_counts = timings[valid_index]
        valid_seconds = buffer_seconds[valid_index]
        segments = self._counter_segments(valid_counts, valid_seconds)
        frame_tas = numpy.asarray(
            getattr(self, 'tas', []), dtype=numpy.float64
        )

        for segment in segments:
            particle_index = valid_index[segment]
            unwrapped, count_delta = self._unwrap_counter(
                timings[particle_index]
            )
            if unwrapped is None:
                continue

            segment_buffer_seconds = buffer_seconds[particle_index]
            segment_buffers = buffer_indices[particle_index]

            has_tas = (
                len(frame_tas) > int(numpy.max(segment_buffers))
            )
            if has_tas:
                particle_tas = frame_tas[segment_buffers]
                has_tas = numpy.all(
                    numpy.isfinite(particle_tas)
                    & (particle_tas > 0.1)
                    & (particle_tas < 500.0)
                )

            if has_tas:
                elapsed = self._counter_elapsed_seconds(
                    count_delta, particle_tas
                )
                offset_x, offset_y = self._smoothed_timing_offset(
                    elapsed, segment_buffer_seconds
                )
                calculated = elapsed + numpy.interp(
                    elapsed, offset_x, offset_y
                )
            else:
                calculated = self._times_from_buffer_anchors(
                    unwrapped,
                    segment_buffers,
                    segment_buffer_seconds,
                )
                if calculated is None:
                    continue

            plausible = (
                numpy.abs(calculated - segment_buffer_seconds)
                <= self.MAX_ANCHOR_DISTANCE_SECONDS
            )
            selected = particle_index[plausible]
            particle_seconds[selected] = calculated[plausible]
            timing_quality[selected] = self.TIMING_BUFFER_COUNTER

        self._replace_backwards_times(
            particle_seconds,
            timing_quality,
            buffer_seconds,
            raw_quality,
        )

        return particle_seconds, timing_quality

    def _apply_hk_counter_timing(
        self,
        timings,
        buffer_seconds,
        valid_buffer,
        particle_seconds,
        timing_quality,
    ):
        """Replace fallback times where valid HK counter anchors are available.

        This is the preferred timing path: HK records provide counter, TAS,
        and UTC anchors while each particle supplies its last-slice counter.
        """
        hk_counts = numpy.asarray(self.hk_counts, dtype=numpy.uint64)
        hk_tas = numpy.asarray(self.hk_tas, dtype=numpy.float64)
        hk_datetimes = numpy.asarray(
            self.hk_datetimes, dtype='datetime64[ns]'
        )
        if not (len(hk_counts) == len(hk_tas) == len(hk_datetimes)):
            raise ValueError('HK timing arrays must have the same length')
        if len(hk_counts) < 2:
            return

        hk_seconds = self._seconds_from_start(hk_datetimes).astype(
            numpy.float64
        )
        valid_hk = (
            numpy.isfinite(hk_seconds)
            & numpy.isfinite(hk_tas)
            & (hk_tas > 0.1)
            & (hk_tas < 500.0)
        )
        hk_counts = hk_counts[valid_hk]
        hk_seconds = hk_seconds[valid_hk]
        hk_tas = hk_tas[valid_hk]
        if len(hk_counts) < 2:
            return

        segments = self._counter_segments(hk_counts, hk_seconds)
        if not segments:
            return

        # Coarse buffer time selects the correct segment after a counter reset.
        segment_id = numpy.full(len(hk_counts), -1, dtype=numpy.int64)
        for number, segment in enumerate(segments):
            segment_id[segment] = number

        anchor_indices = numpy.nonzero(segment_id >= 0)[0]
        time_order = anchor_indices[
            numpy.argsort(hk_seconds[anchor_indices], kind='stable')
        ]
        nearest_sorted = self._nearest_indices(
            hk_seconds[time_order], buffer_seconds
        )
        nearest_hk = time_order[nearest_sorted]
        nearest_segment = segment_id[nearest_hk]
        resolution_m = self.clock_resolution_um * 1.0e-6

        for number, segment in enumerate(segments):
            hk_unwrapped, delta_count = self._unwrap_counter(
                hk_counts[segment]
            )
            if hk_unwrapped is None:
                continue

            hk_time = hk_seconds[segment]
            segment_tas = hk_tas[segment]
            hk_elapsed = self._counter_elapsed_seconds(
                delta_count, segment_tas
            )
            offset_x, offset_y = self._smoothed_timing_offset(
                hk_elapsed, hk_time
            )

            particle_index = numpy.nonzero(
                (nearest_segment == number) & valid_buffer
            )[0]
            if len(particle_index) == 0:
                continue

            local_hk_index = nearest_hk[particle_index] - segment.start
            particle_delta = self._signed_counter_delta(
                timings[particle_index], hk_counts[nearest_hk[particle_index]]
            )
            particle_unwrapped = (
                hk_unwrapped[local_hk_index] + particle_delta
            ).astype(numpy.float64)

            # Reject counters inconsistent with the particle's coarse buffer time.
            count_distance_seconds = (
                numpy.abs(particle_delta).astype(numpy.float64)
                * resolution_m
                / segment_tas[local_hk_index]
            )
            pc_distance_seconds = numpy.abs(
                buffer_seconds[particle_index] - hk_time[local_hk_index]
            )
            max_distance = numpy.maximum(
                self.MAX_ANCHOR_DISTANCE_SECONDS,
                pc_distance_seconds + 2.0,
            )
            plausible = count_distance_seconds <= max_distance
            if not numpy.any(plausible):
                continue

            selected_index = particle_index[plausible]
            selected_count = particle_unwrapped[plausible]
            selected_elapsed = numpy.interp(
                selected_count,
                hk_unwrapped.astype(numpy.float64),
                hk_elapsed,
            )

            # Extrapolation is limited to the endpoint TAS for nearby particles.
            before = selected_count < hk_unwrapped[0]
            after = selected_count > hk_unwrapped[-1]
            selected_elapsed[before] = (
                (selected_count[before] - hk_unwrapped[0])
                * resolution_m
                / segment_tas[0]
            )
            selected_elapsed[after] = (
                hk_elapsed[-1]
                + (selected_count[after] - hk_unwrapped[-1])
                * resolution_m
                / segment_tas[-1]
            )

            selected_offset = numpy.interp(
                selected_elapsed, offset_x, offset_y
            )
            calculated = selected_elapsed + selected_offset
            close_to_buffer = numpy.abs(
                calculated - buffer_seconds[selected_index]
            ) <= self.MAX_ANCHOR_DISTANCE_SECONDS
            selected_index = selected_index[close_to_buffer]
            particle_seconds[selected_index] = calculated[close_to_buffer]
            timing_quality[selected_index] = self.TIMING_HK_COUNTER

    def _calculate_particle_times(self, timings, buffer_indices):
        """Calculate particle times from probe counters and available anchors.

        Probe counters provide short-period timing. The nominally 1-Hz HK
        timestamps are preferred UTC anchors. If they are unavailable, 4-kB
        buffer timestamps anchor the probe-counter timeline instead.
        """
        timings = numpy.asarray(timings, dtype=numpy.uint64)
        buffer_indices = numpy.asarray(buffer_indices, dtype=numpy.int64)
        if timings.shape != buffer_indices.shape:
            raise ValueError('timings and buffer_indices must have the same shape')

        frame_seconds = self._seconds_from_start(self.datetimes).astype(numpy.float64)
        valid_buffer = (
            (buffer_indices >= 0)
            & (buffer_indices < len(frame_seconds))
        )
        buffer_seconds = numpy.zeros(len(timings), dtype=numpy.float64)
        buffer_seconds[valid_buffer] = frame_seconds[
            buffer_indices[valid_buffer]
        ]
        # Establish a complete baseline before attempting the preferred HK
        # calculation. Every particle therefore retains a usable fallback.
        particle_seconds, timing_quality = self._buffer_counter_fallback(
            timings,
            buffer_indices,
            buffer_seconds,
            valid_buffer,
        )
        fallback_seconds = particle_seconds.copy()
        fallback_quality = timing_quality.copy()

        if len(timings) == 0:
            sec, ns = self._seconds_to_sec_ns(particle_seconds)
            return sec, ns, timing_quality

        # Prefer HK-anchored times, but retain the buffer-derived fallback for
        # missing, malformed, or implausible HK counter records.
        self._apply_hk_counter_timing(
            timings,
            buffer_seconds,
            valid_buffer,
            particle_seconds,
            timing_quality,
        )
        self._replace_backwards_times(
            particle_seconds,
            timing_quality,
            fallback_seconds,
            fallback_quality,
        )

        sec, ns = self._seconds_to_sec_ns(particle_seconds)
        return sec, ns, timing_quality

    def _finalize_particle_times(self, spiffile, inst_name):
        """Replace provisional buffer times with global counter-derived times."""
        instgrp = spiffile.instgrps[inst_name]
        coregrp = instgrp['core']
        if 'clock_counts' not in coregrp.variables:
            return

        timings = numpy.asarray(coregrp['clock_counts'][:], dtype=numpy.uint64)
        buffer_indices = numpy.asarray(
            coregrp['buffer_index'][:], dtype=numpy.int64
        )
        sec, ns, quality = self._calculate_particle_times(
            timings, buffer_indices
        )
        coregrp['image_sec'][:] = sec
        coregrp['image_ns'][:] = ns
        if 'timing_quality' in coregrp.variables:
            coregrp['timing_quality'][:] = quality
            coregrp['timing_quality'].setncatts(
                dict(self.AUX_ATTRS['timing_quality'])
            )
        coregrp['clock_counts'].setncatts(
            dict(self.AUX_ATTRS['clock_counts'])
        )
        coregrp['image_sec'].setncattr(
            'ancillary_variables', 'timing_quality clock_counts'
        )

        instgrp.setncattr(
            'particle_timing_method',
            '48-bit probe-counter timing with automatic HK timestamp, '
            '4-kB buffer timestamp, and raw buffer-time fallback',
        )
        instgrp.setncattr(
            'particle_timestamp_reference',
            'Time of the last slice in the particle image',
        )
        instgrp.setncattr('clock_resolution_um', self.clock_resolution_um)
        spiffile.rootgrp.sync()

        hk_counter = int(numpy.count_nonzero(
            quality == self.TIMING_HK_COUNTER
        ))
        buffer_counter = int(numpy.count_nonzero(
            quality == self.TIMING_BUFFER_COUNTER
        ))
        buffer_only = int(numpy.count_nonzero(
            quality == self.TIMING_BUFFER_ONLY
        ))
        invalid = int(numpy.count_nonzero(
            quality == self.TIMING_INVALID_BUFFER
        ))
        print(
            f'{inst_name} timing: {hk_counter} HK-counter, '
            f'{buffer_counter} buffer-counter, {buffer_only} buffer-only, '
            f'{invalid} invalid.'
        )


    def extract_images(self, buffer):
        """
        Implementation of the abstract extract_images method.
        Slices the buffer into individual Image objects.
        Returns separate Images objects for H and V channels to match SPECFile structure.

        Workers write provisional buffer timestamps. After all ordered chunks
        are written, process_file replaces them with globally integrated probe
        counter times so chunk boundaries cannot reset timing state.

        History:
            - 2026-07-15: Yongjie Huang, added probe-counter timing.
            - 2026-01-11: Yongjie Huang, first implementation.
        """
        h_imgs = Images(self.aux_channels)
        v_imgs = Images(self.aux_channels)

        frame_seconds = self._seconds_from_start(self.datetimes).astype(
            numpy.float64
        )

        # Process each channel with vectorization
        for channel_name, imgs_obj in [('h', h_imgs), ('v', v_imgs)]:
            img_list = buffer.get(channel_name, [])
            if not img_list:
                continue

            # 1. First pass: Collect all data into lists
            raw_data_list = []
            timings_list = []
            buf_indices_list = []

            # Filter valid images and extract data
            for img_dict in img_list:
                if 'data' in img_dict and len(img_dict['data']) > 0:
                     raw_data_list.append(img_dict)
                     timings_list.append(img_dict.get('time', 0))
                     buf_indices_list.append(img_dict.get('buffer_index', 0))

            if not raw_data_list:
                continue

            buf_indices_arr = numpy.asarray(buf_indices_list, dtype=numpy.int64)
            valid_buffer = (
                (buf_indices_arr >= 0)
                & (buf_indices_arr < len(frame_seconds))
            )
            particle_seconds = numpy.zeros(
                len(buf_indices_arr), dtype=numpy.float64
            )
            particle_seconds[valid_buffer] = frame_seconds[
                buf_indices_arr[valid_buffer]
            ]
            provisional_quality = numpy.where(
                valid_buffer,
                self.TIMING_BUFFER_ONLY,
                self.TIMING_INVALID_BUFFER,
            )
            sec_array, ns_array = self._seconds_to_sec_ns(particle_seconds)

            # 3. Populate Images object
            for i, img_dict in enumerate(raw_data_list):
                try:
                    image_data = numpy.ascontiguousarray(img_dict['data'], dtype=numpy.uint8)

                    imgs_obj.image.append(image_data)
                    imgs_obj.ns.append(int(ns_array[i]))
                    imgs_obj.sec.append(int(sec_array[i]))
                    imgs_obj.length.append(len(image_data) // 128)
                    imgs_obj.clock_counts.append(timings_list[i])
                    imgs_obj.timing_quality.append(
                        int(provisional_quality[i])
                    )
                    imgs_obj.overload_flag.append(
                        int(img_dict.get('overload_flag', 0))
                    )

                    buf_idx = buf_indices_list[i]
                    imgs_obj.buffer_index.append(buf_idx)

                    for channel in ('tas', 'user_temp', 'ps_temp'):
                        values = getattr(self, channel, None)
                        value = (
                            values[buf_idx]
                            if (
                                valid_buffer[i]
                                and values is not None
                                and len(values) > buf_idx
                            )
                            else numpy.nan
                        )
                        getattr(imgs_obj, channel).append(value)

                except (ValueError, TypeError):
                    pass

        return h_imgs, v_imgs
