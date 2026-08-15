# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

# Format Constants
cdef unsigned short SYNC_2S = 12883           # 0x3253
cdef unsigned short SYNC_NL = 0x4E4C          # disk bytes 0x4C 0x4E
cdef unsigned short RLE_FULL_SHADED = 0x4000  # 0x4000
cdef unsigned short RLE_UNCOMPRESSED = 0x7FFF # 0x7FFF

@cython.boundscheck(False)
@cython.wraparound(False)
def decode_frame(
    const unsigned short[:] record,
    const unsigned short[:] record_next,
    bint next_record_exists,
    dict chunk_state,
    int frame_idx,
    int start_idx=0
):
    """
    Optimized Cython implementation of process_frame_with_state.
    """
    cdef:
        int idx = start_idx
        int limit = record.shape[0]
        int next_start_idx = 0
        int i, j, k
        unsigned short word, nh_raw, nv_raw, pid, slices
        unsigned short nh, nv, n_words, is_multi_packet, overload_flag
        int data_start, data_end, words_needed
        unsigned long long timing = 0
        unsigned short val
        int startslice, num_shaded, num_clear

        # Temporary buffers/vars
        list full_packet_data = [] # List of unsigned shorts
        list h_images = []
        list v_images = []
        dict pending_h = chunk_state['pending_h']
        dict pending_v = chunk_state['pending_v']
        dict state
        dict pending
        bint is_horiz

        # Decompression vars
        list img_decomp
        list slice_decomp
        int non_compressed
        bytearray final_buffer
        np.ndarray final_data

    while idx < limit:
        word = record[idx]

        if word == SYNC_2S:
            # Header Check
            if idx + 5 <= limit:
                nh_raw = record[idx+1]
                nv_raw = record[idx+2]
                pid = record[idx+3]
                slices = record[idx+4]
            else:
                # Boundary case - stitch header
                if not next_record_exists:
                    break
                # Construct temp header manually
                # Not optimizing this edge case heavily as it's rare per frame (max 1)
                words_needed = 5 - (limit - idx)
                if words_needed > record_next.shape[0]:
                    break
                temp_header = [record[k] for k in range(idx, limit)]
                for k in range(words_needed):
                    temp_header.append(record_next[k])

                nh_raw = temp_header[1]
                nv_raw = temp_header[2]
                pid = temp_header[3]
                slices = temp_header[4]

            # Word counts
            nh = nh_raw & 0x0FFF
            nv = nv_raw & 0x0FFF
            n_words = nh if nh > 0 else nv

            if n_words == 0:
                idx += 1
                continue

            is_horiz = (nh > 0)
            is_multi_packet = ((nh_raw if is_horiz else nv_raw) & 0x1000) >> 12
            overload_flag = ((nh_raw if is_horiz else nv_raw) & 0x8000) >> 15

            data_start = idx + 5
            data_end = data_start + n_words

            # Extract Packet Data
            if data_end > limit:
                if next_record_exists:
                    words_needed = data_end - limit
                    if words_needed > record_next.shape[0]:
                        print(
                            f'Warning: Fast 2DS packet at frame {frame_idx}, '
                            f'word {idx} spans more than two frames '
                            f'(n_words={n_words}); packet skipped.'
                        )
                        break
                    next_start_idx = words_needed

                    if data_start < limit:
                        full_packet_data = [record[k] for k in range(data_start, limit)]
                        for k in range(words_needed):
                            full_packet_data.append(record_next[k])
                    else:
                        full_packet_data = [record_next[k - limit] for k in range(data_start, data_end)]
                else:
                    break # EOF logic
            else:
                full_packet_data = [record[k] for k in range(data_start, data_end)]

            # --- Decoding Logic ---
            pending = pending_h if is_horiz else pending_v

            # Using Python dicts for state is the bottleneck here potentially
            # But converting the entire state machine to C structs is complex
            # We speed up the inner RLE loop primarily
            if pid not in pending:
                pending[pid] = {'img_decomp': [], 'slice_decomp': [], 'non_compressed': 0}
            state = pending[pid]

            img_decomp = state['img_decomp']
            slice_decomp = state['slice_decomp']
            non_compressed = state['non_compressed']

            # Handle Timing / Payload Split
            payload_len = len(full_packet_data)
            if is_multi_packet:
                timing = 0
            elif payload_len >= 3:
                 timing = (full_packet_data[payload_len-3] |
                           (full_packet_data[payload_len-2] << 16) |
                           (full_packet_data[payload_len-1] << 32))
                 # Remove last 3
                 payload_len -= 3
            else:
                timing = 0

            # Process Payload
            for i in range(payload_len):
                val = full_packet_data[i]

                if non_compressed > 0:
                    # Raw bitmap mode: 16 pixels
                    # reversed(bin(val)[2:].zfill(16)).
                    # 1=Clear(input), 0=Shaded(input).
                    # Output: 0 for Clear, 1 for Shaded. => Output = 1 - input_bit
                    # LSB is first pixel in time/stream?
                    # Python Code: bin_line = [-1 * (int(n) - 1) for n in bin(val)[2:].zfill(16)[::-1]]
                    # Example: val=1 (0...01). bin='1'. zfill='0...01'. reverse='10...0'.
                    # First char '1' (clear) -> -1*(1-1)=0 (clear).
                    # Second char '0' (shaded) -> -1*(0-1)=1 (shaded).
                    # So LSB (bit 0) of val matches the END of the string?
                    # Python [::-1] means we process last char first.
                    # Last char of zfill'd string is LSB.
                    # So LSB corresponds to the first element in the list?
                    # Yes.

                    for j in range(16):
                        # Extract bit j
                        bit = (val >> j) & 1
                        # bit 1 (clear) -> 0. bit 0 (shaded) -> 1.
                        slice_decomp.append(1 - bit)

                    non_compressed -= 1

                    if non_compressed == 0 and len(slice_decomp) > 0:
                         # Pad
                         rem = len(slice_decomp) % 128
                         if rem > 0:
                             for k in range(128 - rem):
                                 slice_decomp.append(0)
                         img_decomp.extend(slice_decomp)
                         slice_decomp[:] = []

                elif val == RLE_UNCOMPRESSED:
                    if len(slice_decomp) > 0:
                        rem = len(slice_decomp) % 128
                        if rem > 0:
                            for k in range(128 - rem):
                                slice_decomp.append(0)
                        img_decomp.extend(slice_decomp)
                        slice_decomp[:] = []
                    non_compressed = 8

                elif val == RLE_FULL_SHADED:
                     if len(slice_decomp) > 0:
                        rem = len(slice_decomp) % 128
                        if rem > 0:
                            for k in range(128 - rem):
                                slice_decomp.append(0)
                        img_decomp.extend(slice_decomp)
                        slice_decomp[:] = []
                     # Add 128 ones
                     for k in range(128):
                         img_decomp.append(1)

                else:
                    # RLE Word
                    startslice = (val >> 14) & 1
                    num_shaded = (val >> 7) & 0x7F
                    num_clear = val & 0x7F

                    if startslice == 1 and len(slice_decomp) > 0:
                        rem = len(slice_decomp) % 128
                        if rem > 0:
                             for k in range(128 - rem):
                                 slice_decomp.append(0)
                        img_decomp.extend(slice_decomp)
                        slice_decomp[:] = []

                    for k in range(num_clear):
                        slice_decomp.append(0)
                    for k in range(num_shaded):
                        slice_decomp.append(1)

            # Write back state
            state['non_compressed'] = non_compressed

            # Finalize Image
            if not is_multi_packet:
                 if len(slice_decomp) > 0:
                     rem = len(slice_decomp) % 128
                     if rem > 0:
                          for k in range(128 - rem):
                               slice_decomp.append(0)
                     img_decomp.extend(slice_decomp)

                 # bytearray performs the list conversion efficiently and owns
                 # the compact NumPy buffer. Invert internal shaded/clear bits.
                 final_buffer = bytearray(img_decomp)
                 final_data = np.frombuffer(final_buffer, dtype=np.uint8)
                 np.bitwise_xor(final_data, 1, out=final_data)

                 img_result = {
                     'id': pid,
                     'slices': slices,
                     'time': timing,
                     'data': final_data,
                     'buffer_index': frame_idx,
                     'overload_flag': overload_flag,
                 }

                 if is_horiz:
                     h_images.append(img_result)
                 else:
                     v_images.append(img_result)

                 del pending[pid]

            # Update loop
            if data_end > limit:
                idx = limit + 1
            else:
                idx = data_end

        elif word == SYNC_NL:
            idx += 1
        else:
            idx += 1

    return {'h': h_images, 'v': v_images, 'next_idx': next_start_idx}
