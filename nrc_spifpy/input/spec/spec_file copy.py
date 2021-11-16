import datetime

import numpy as np

from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.images import Images

import numba as nb
from numba import jit
from numba.experimental import jitclass
from numba import types, typed

from tqdm import tqdm

file_buffer_template = np.dtype([
    ('year', 'u2'),
    ('month', 'u2'),
    ('weekday', 'u2'),
    ('day', 'u2'),
    ('hour', 'u2'),
    ('minute', 'u2'),
    ('second', 'u2'),
    ('ms', 'u2'),
    ('data', '(2048, )u2'),
    ('discard', 'u2')
])

record_metadata_template = np.dtype([
    ("n_h","u2"),
    ("timing_h","u2"),
    ("mismatch_h","u2"),
    ("fifo_h","u2"),
    ("overload_h","u2"),
    ("n_v","u2"),
    ("timing_v","u2"),
    ("mismatch_v","u2"),
    ("fifo_v","u2"),
    ("overload_v","u2"),
    ("particle_count", "u2"),
    ("num_slices", "u2"),
    ("timing_word_upper_h","u2"),
    ("timing_word_lower_h","u2"),
    ("timing_word_upper_v","u2"),
    ("timing_word_lower_v","u2")
])

img_template = np.dtype([
    ("metadata", record_metadata_template),
    ("img_h", "object"),
    ("img_v", "object")
])

housekeeping_template = np.dtype([
    ("packet_id_word","u2"),
    ("hz_elem_0_voltage","u2"),
    ("hz_elem_64_voltage","u2"),
    ("hz_elem_127_voltage","u2"),
    ("v_elem_0_voltage","u2"),
    ("v_elem_64_voltage","u2"),
    ("v_elem_127_voltage","u2"),
    ("raw_pos_power_supply","u2"),
    ("raw_neg_power_supply","u2"),
    ("hz_arm_tx_temp","u2"),
    ("hz_arm_rx_temp","u2"),
    ("v_arm_tx_temp","u2"),
    ("v_arm_rx_temp","u2"),
    ("hz_tip_tx_temp","u2"),
    ("hz_tip_rx_temp","u2"),
    ("rear_opt_bridge_temp","u2"),
    ("dsp_board_temp","u2"),
    ("forward_vessel_temp","u2"),
    ("hz_laser_temp","u2"),
    ("v_laser_temp","u2"),
    ("front_plate_temp","u2"),
    ("power_supply_temp","u2"),
    ("minus_5V_supply","u2"),
    ("plus_5V_supply","u2"),
    ("can_internal_pressure","u2"),
    ("hz_elem_21_voltage","u2"),
    ("hz_elem_42_voltage","u2"),
    ("hz_elem_85_voltage","u2"),
    ("hz_elem_106_voltage","u2"),
    ("v_elem_21_voltage","u2"),
    ("v_elem_42_voltage","u2"),
    ("v_elem_85_voltage","u2"),
    ("v_elem_106_voltage","u2"),
    ("num_v_particles_detected","u2"),
    ("num_h_particles_detected","u2"),
    ("heater_outputs","u2"),
    ("h_laser_drive","u2"),
    ("v_laser_drive","u2"),
    ("hz_masked_bits","u2"),
    ("v_masked_bits","u2"),
    ("num_stereo_particles_detected","u2"),
    ("num_t_word_mismatch","u2"),
    ("num_slice_count_mismatch","u2"),
    ("num_hz_ov_periods","u2"),
    ("num_v_ov_periods","u2"),
    ("compression_conf","u2"),
    ("num_empty_fifo","u2"),
    ("spare_2","u2"),
    ("spare_3","u2"),
    ("tas","int32"),
    ("timing_word_1","int32")
])

mask_template = np.dtype([
    ("flag","u2"),
    ("timing_word", "int32"),
    ("hz_mask_byte_1", "u2"),
    ("hz_mask_byte_2", "u2"),
    ("hz_mask_byte_3", "u2"),
    ("hz_mask_byte_4", "u2"),
    ("hz_mask_byte_5", "u2"),
    ("hz_mask_byte_6", "u2"),
    ("hz_mask_byte_7", "u2"),
    ("hz_mask_byte_8", "u2"),
    ("v_mask_byte_1", "u2"),
    ("v_mask_byte_2", "u2"),
    ("v_mask_byte_3", "u2"),
    ("v_mask_byte_4", "u2"),
    ("v_mask_byte_5", "u2"),
    ("v_mask_byte_6", "u2"),
    ("v_mask_byte_7", "u2"),
    ("v_mask_byte_8", "u2"),
    ("timing_word_start", "int32"),
    ("timing_word_end", "int32")
])

DATA_FLAG = (ord('2') << 8) + ord('S')
HK_FLAG = (ord('H') << 8) + ord('K')
MK_FLAG = (ord('M') << 8) + ord('K')

WORD_H_OFFSET = 1
WORD_V_OFFSET = 2
PC_OFFSET = 3
NUM_SLICE_OFFSET = 4

aux_params = [
    "num_words",
    "timing_flag",
    "mismatch_flag",
    "fifo_flag",
    "overload_flag",
    "particle_count",
    "num_slices",
    "timing_word_upper",
    "timing_word_lower",
    "tas"
]

class SPECFile(BinaryFile):
    """ Class representing monoscale binary format for SPEC instruments.
    Implements methods specific to decompressing and processing images from
    this type of binary file.
    Parameters
    ----------
    filename : str
        Filename of current file.
    inst_name : str
        Name of current instrument.
    """

    def __init__(self, filename, inst_name, resolution):
        super().__init__(filename, inst_name, resolution)
        self.diodes = 128
        self.file_dtype = file_buffer_template
        self.aux_channels = aux_params


    def calc_buffer_datetimes(self):
        """ Calculates datetimes from buffers read in from file and sets
        to datetimes class attribute.
        """
        self.datetimes = [datetime.datetime(d['year'],
                                            d['month'],
                                            d['day'],
                                            d['hour'],
                                            d['minute'],
                                            d['second'],
                                            d['ms'] * 1000) for d in self.data]

        self.timestamps = [(x - self.start_date).total_seconds() for x in self.datetimes]

    def process_file(self, spiffile, processors):
        spiffile.set_start_date(self.start_date.strftime('%Y-%m-%d %H:%M:%S %z'))

        if self.name == '2DS':
            spiffile.create_inst_group(self.name + '-V')
            spiffile.create_inst_group(self.name + '-H')
            spiffile.set_filenames_attr(self.name + '-V', self.filename)
            spiffile.set_filenames_attr(self.name + '-H', self.filename)
        else:
            spiffile.create_inst_group(self.name)
            spiffile.set_filenames_attr(self.name, self.filename)

        spiffile.write_buffer_info(self.start_date, self.datetimes)

        pbar1 = tqdm(desc='Processing frames',
                     total=len(self.data),
                     unit='frame')

        for i, buffer in enumerate(self.data):

            buffer = Buffer(i, self.timestamps[i], buffer['data'])

            #merged_buffers_h, merged_buffers_v = self.merge_buffers(buffer)

            #self.write_data(spiffile, merged_buffers_h, merged_buffers_v)
            self.write_data(spiffile, buffer.image_containers['h'], buffer.image_containers['v'])

            pbar1.update(1)

    def get_chunk_intervals(self):
        data_len = len(self.data)

        chunk_intervals = [i for i in range(data_len, 500)]

        if chunk_intervals[-1] < data_len:
            chunk_intervals.append(data_len)
        return chunk_intervals

    def merge_buffers(self, buffers):
        merged_buffers_h = Images(aux_params)
        merged_buffers_v = Images(aux_params)

        for buffer in buffers:
            merged_buffers_h.extend(buffer.image_containers['h'])
            merged_buffers_v.extend(buffer.image_containers['v'])

        return merged_buffers_h, merged_buffers_v

    def write_data(self, spiffile, h_p, v_p):
        """ Called each time number of unsaved processed images exceeds
        10,000. Writes H and V images to SPIF NetCDF file.
        Parameters
        ----------
        spiffile : SPIFFile object
            SPIFFile object of current SPIF NetCDF output file
        h_p : Images object
            Images object containing processed H array image info.
        v_p : Images object
            Images object containing processed V array image info.
        """

        if self.name == '2DS':
            if len(h_p) > 0:
                h_p.conv_to_array(self.diodes)
                spiffile.write_images(self.name + '-H', h_p)
            v_suffix = '-V'
        else:
            v_suffix = ''

        if len(v_p) > 0:
            v_p.conv_to_array(self.diodes)
            spiffile.write_images(self.name + v_suffix, v_p)

class Buffer:

    def __init__(self, buffer_id, buffer_ts, buffer):
        self.buffer_id = buffer_id
        self.buffer_sec = self.get_buffer_sec(buffer_ts)
        self.buffer_ns = self.get_buffer_ns(buffer_ts)
        self.buffer = buffer
        self.image_containers = self.generate_image_containers()
        self.housekeeping_frames = []
        self.mask_frames = []

        self.alloc_frames()

    def generate_image_containers(self):
        return {
            "h":Images(aux_params),
            "v":Images(aux_params)
        }

    def get_buffer_sec(self, buffer_ts):
        return int(np.floor(buffer_ts))

    def get_buffer_ns(self, buffer_ts):
        return int( (buffer_ts - np.floor(buffer_ts))*1e9 )

    def alloc_frames(self):
        
        buffer_idx = 0

        while buffer_idx < (2048 - 53):
                flag = self.buffer[buffer_idx]

                if flag == DATA_FLAG:
                    raw_image = RawImageContainer(
                        buffer_idx,
                        self.buffer_sec, 
                        self.buffer_ns, 
                        self.buffer
                    )

                    self.image_containers['h'].extend(
                        self.get_prepped_img_h(raw_image)
                    )

                    self.image_containers['v'].extend(
                        self.get_prepped_img_v(raw_image)
                    )

                    buffer_idx += raw_image.frame_len

                elif flag == HK_FLAG:
                    self.housekeeping_frames.append(np.array([self.buffer[buffer_idx : buffer_idx + 53]], dtype = housekeeping_template))
                    buffer_idx += 53
                elif flag == MK_FLAG:
                    self.mask_frames.append(np.array([self.buffer[buffer_idx : buffer_idx + 23]], dtype = mask_template))
                    buffer_idx += 23
                else:
                    buffer_idx += 1

    def get_prepped_img_h(self, img_container_h):
        prepped_img_h = Images(aux_params)

        prepped_img_h.sec = [img_container_h.buffer_sec]
        prepped_img_h.ns = [img_container_h.buffer_ns]
        prepped_img_h.image = [img_container_h.image_h]
        prepped_img_h.length = [len(img_container_h.image_h)/128]
        prepped_img_h.buffer_index = [img_container_h.buffer_idx]
        prepped_img_h.num_words = [img_container_h.metadata['n_h']]
        prepped_img_h.timing_flag = [img_container_h.metadata['timing_h']]
        prepped_img_h.mismatch_flag = [img_container_h.metadata['mismatch_h']]
        prepped_img_h.fifo_flag = [img_container_h.metadata['fifo_h']]
        prepped_img_h.overload_flag = [img_container_h.metadata['overload_h']]
        prepped_img_h.particle_count = [img_container_h.metadata['particle_count']]
        prepped_img_h.num_slices = [img_container_h.metadata['num_slices']]
        prepped_img_h.timing_word_upper = [img_container_h.metadata['timing_word_upper_h']]
        prepped_img_h.timing_word_lower = [img_container_h.metadata['timing_word_lower_h']]
        prepped_img_h.tas = [100.0]

        return prepped_img_h

    def get_prepped_img_v(self, img_container_v):
        prepped_img_v = Images(aux_params)

        prepped_img_v.sec = [img_container_v.buffer_sec]
        prepped_img_v.ns = [img_container_v.buffer_ns]
        prepped_img_v.image = [img_container_v.image_v]
        prepped_img_v.length = [len(img_container_v.image_v)/128]
        prepped_img_v.buffer_index = [img_container_v.buffer_idx]
        prepped_img_v.num_words = [img_container_v.metadata['n_v']]
        prepped_img_v.timing_flag = [img_container_v.metadata['timing_v']]
        prepped_img_v.mismatch_flag = [img_container_v.metadata['mismatch_v']]
        prepped_img_v.fifo_flag = [img_container_v.metadata['fifo_v']]
        prepped_img_v.overload_flag = [img_container_v.metadata['overload_v']]
        prepped_img_v.particle_count = [img_container_v.metadata['particle_count']]
        prepped_img_v.num_slices = [img_container_v.metadata['num_slices']]
        prepped_img_v.timing_word_upper = [img_container_v.metadata['timing_word_upper_v']]
        prepped_img_v.timing_word_lower = [img_container_v.metadata['timing_word_lower_v']]
        prepped_img_v.tas = [100.0]

        return prepped_img_v
        
"""
@jitclass([
    ("buffer_idx", nb.types.int32),
    ("buffer_sec", nb.types.int32),
    ("buffer_ns", nb.types.int32),
    ("metadata", nb.from_dtype(record_metadata_template)[:]),
    ("image_h", nb.types.boolean[:]),
    ("image_v", nb.types.boolean[:]),
])
"""
class RawImageContainer:

    def __init__(self, buffer_idx, buffer_sec, buffer_ns, buffer):
        self.buffer_idx = buffer_idx

        self.buffer_sec = buffer_sec
        self.buffer_ns = buffer_ns

        self.metadata = self.get_metadata(buffer)

        self.image_h = self.get_image_h(buffer)
        self.image_v = self.get_image_v(buffer)

    def get_metadata(self, buffer):
        
        metadata = Metadata_Processing_Helper().process_metadata(
            buffer[self.buffer_idx + WORD_H_OFFSET], 
            buffer[self.buffer_idx + WORD_V_OFFSET],
            buffer[self.buffer_idx + PC_OFFSET],
            buffer[self.buffer_idx + NUM_SLICE_OFFSET]
        )
        
        return metadata

    def get_image_h(self, buffer):
        image_h = buffer[self.h_bounds]

        if not (self.metadata['timing_h'][0]) and len(image_h) >= 2:
            self.metadata['timing_word_upper_h'][0] = image_h[-2]
            self.metadata['timing_word_lower_h'][0] = image_h[-1]

            image_h = image_h[:-2]
        else:
            self.metadata['timing_word_upper_h'][0] = 0
            self.metadata['timing_word_lower_h'][0] = 0
        breakpoint()
        decompressed_image_h = self.decompress_image(image_h)

        if (self.metadata['n_h'] > 2) and len(decompressed_image_h) == 0: 
            breakpoint()

        return decompressed_image_h

    def get_image_v(self, buffer):
        image_v = buffer[self.v_bounds]

        if not (self.metadata['timing_v'][0]) and len(image_v) >= 2:
            self.metadata['timing_word_upper_v'][0] = image_v[-2]
            self.metadata['timing_word_lower_v'][0] = image_v[-1]

            image_v = image_v[:-2]
        else:
            self.metadata['timing_word_upper_v'][0] = 0
            self.metadata['timing_word_lower_v'][0] = 0

        decompressed_image_v = self.decompress_image(image_v)

        #if (self.metadata['n_v'] > 2): 
        #    breakpoint()

        return decompressed_image_v

    @property
    def h_bounds(self):
        return slice(
            self.buffer_idx + 5,
            self.buffer_idx + 5 + self.metadata['n_h'][0]
        )

    @property
    def v_bounds(self):
        return slice(
            self.buffer_idx + 5 + self.metadata['n_h'][0], 
            self.buffer_idx + 5 + self.metadata['n_h'][0] + self.metadata['n_v'][0]
        )

    @property
    def frame_len(self):
        # 5 = 1 flag word, and the rest of the metadata
        return 5 + self.metadata['n_h'][0] + self.metadata['n_v'][0]

    def decompress_image(self, compressed_image):
        """ Decompresses image.
        Parameters
        ----------
        img : array
            Compressed image data.
        Returns
        -------
        array
            Array of decompressed image.
        """

        encoded_word_type = np.dtype([
            ("is_image_slice", "u2"),
            ("is_start_slice", "u2"),
            ("num_shaded", "u2"),
            ("num_clear", "u2")
        ])

        decoded_images = np.zeros(len(compressed_image), dtype = encoded_word_type)

        for i, word in enumerate(compressed_image):
            if word == 0x7fff:
                decoded_images['is_image_slice'][i] = 1
                decoded_images['is_start_slice'][i] = 1
                decoded_images['num_clear'][i] = 128
                decoded_images['num_shaded'][i] = 0
            elif word == 0x4000:
                decoded_images['is_image_slice'][i] = 1
                decoded_images['is_start_slice'][i] = 1
                decoded_images['num_clear'][i] = 0
                decoded_images['num_shaded'][i] = 128
            else:
                decoded_images['is_image_slice'][i] = (word & 2**15) >> 15
                decoded_images['is_start_slice'][i] = (word & 2**14) >> 14
                decoded_images['num_shaded'][i] = (word & 0b0011111110000000) >> 7
                decoded_images['num_clear'][i]  = (word & 0b0000000001111111)

        decompressed_image = [0]

        return decompressed_image


    def decompress_image_old(self, img):
        """ Decompresses image.
        Parameters
        ----------
        img : array
            Compressed image data.
        Returns
        -------
        array
            Array of decompressed image.
        """

        img_decomp = [int(x) for x in range(0)]
        slice_decomp = [int(x) for x in range(0)]
        for line in img:
            if line == 32767:  # special case of 0x7fff
                img_decomp.extend([0] * 128)
            elif line == 16384:  # special case of 0x4000
                img_decomp.extend([1] * 128)
            else:
                #timeslice = (line & (2 ** 15)) >> 15
                timeslice = line > 32768
                startslice = (line & (2 ** 14)) >> 14
                num_shaded = (line & 16256) >> 7
                num_clear = (line & 127)

                if timeslice == 0:
                    if startslice == 1:
                        if len(slice_decomp) % 128 > 0:
                            slice_decomp.extend([0] * (128 - (len(slice_decomp) % 128)))
                        img_decomp.extend(slice_decomp)
                        slice_decomp = [int(x) for x in range(0)]

                    slice_decomp.extend([0] * num_clear)
                    slice_decomp.extend([1] * num_shaded)

        

        img = np.logical_not(np.array(img_decomp))

        return img

@jitclass([
    ("metadata", nb.from_dtype(record_metadata_template)[:])
])
class Metadata_Processing_Helper:
    """
    This is for words 2 in a particle frame
    NH (Word 2)
    -----------------------------------------------------------
    Bits 0–11 Number of horizontal words–Includes Timing Words if present
    Bit 12 – 1 = Timing Words not found 
    Bit 13 – Timing Word mismatch
    Bit 14 — FIFO Empty (means the next particle was cut off)
    Bit 15 – The last two words of the horizontal data record are overload timing words
    NV (Word 3)
    -------------------------------------------------------------
    Bits 0 –11 Number of vertical words–Includes Timing Words if not same as  the  horizontal Timing Word and the TW were found.
    Bit 12 –1 = Timing Words not found 
    Bit 13 –Timing Word mismatch 
    Bit 14-FIFO Empty before timing word found
    Bit 15 –The last two words of the vertical data record are overload timing words
    """

    def __init__(self) -> None:
        self.metadata = np.empty(1, dtype = record_metadata_template)

    def process_metadata(self, word_h, word_v, word_pc, word_num_slices):

        self.metadata['n_h'][:] = self.num_words(word_h)
        self.metadata['timing_h'][:] = self.timing_words_not_found(word_h)
        self.metadata['mismatch_h'][:] = self.timing_word_mismatch(word_h)
        self.metadata['fifo_h'][:] = self.fifo_empty(word_h)
        self.metadata['overload_h'][:] = self.overload_timing_words_exist(word_h)
        self.metadata['n_v'][:] = self.num_words(word_v)
        self.metadata['timing_v'][:] = self.timing_words_not_found(word_v)
        self.metadata['mismatch_v'][:] = self.timing_word_mismatch(word_v)
        self.metadata['fifo_v'][:] = self.fifo_empty(word_v)
        self.metadata['overload_v'][:] = self.overload_timing_words_exist(word_v)
        self.metadata['particle_count'][:] = word_pc
        self.metadata['num_slices'][:] = word_num_slices

        return self.metadata

    def num_words(self, word):
        # Bit masking out of a 16-bit number
        # to only get the 12 bit component
        return word & 0b0000111111111111

    def timing_words_not_found(self, word):
        # Bitmask to get 12th bit only, then bit shift right
        # 12 spots to keep only that bit
        return (word & 0b0001000000000000) >> 12

    def timing_word_mismatch(self, word):
        # Bitmask to get 13th bit only, then bit shift right
        # 13 spots to keep only that bit
        return (word & 0b0010000000000000) >> 13

    def fifo_empty(self, word):
        # Bitmask to get 14th bit only, then bit shift right
        # 14 spots to keep only that bit
        return (word & 0b0100000000000000) >> 14

    def overload_timing_words_exist(self, word):
        # Bitmask to get 15th bit only, then bit shift right
        # 15 spots to keep only that bit
        return (word & 0b1000000000000000) >> 15

class HouseKeepingPacket:

    def __init__(self) -> None:
        pass

class MaskPacket:

    def __init__(self) -> None:
        pass