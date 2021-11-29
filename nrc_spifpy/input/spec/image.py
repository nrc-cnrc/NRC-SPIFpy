import numpy as np
import numba as nb
from numba import types, typed, typeof
from numba import jit
from numba.experimental import jitclass

from nrc_spifpy.spif import TIME_CHUNK

# The size of the metadata in a particle record
# Word 1 = Flag 2S
# Word 2 = word for h image metadata
# Word 3 = word for v image metadata
# Word 4 = word for particle count
# Word 5 = word for number of slices

METADATA_LENGTH = 5

# Offsets to find specific metadata in an image record

WORD_H_OFFSET = 1
WORD_V_OFFSET = 2
WORD_PC_OFFSET = 3
WORD_NUM_SLICE_OFFSET = 4

# Easier to define here than to have to flip 1 and 0 in the code

SHADED_VAL = 0
CLEAR_VAL = 1

# Useful datatypes

decoded_word_type = np.dtype([
            ("is_image_slice", "u2"),
            ("is_start_slice", "u2"),
            ("num_shaded", "u2"),
            ("num_clear", "u2")
        ])

class ImageMetadataContainer:

    def __init__(self):
        self.buffer_idx = 0

        self.n_h = 0
        self.timing_h = 0
        self.mismatch_h = 0
        self.fifo_h = 0
        self.overload_h = 0

        self.n_v = 0
        self.timing_v = 0
        self.mismatch_v = 0
        self.fifo_v = 0
        self.overload_v = 0

        self.particle_count = 0
        self.num_slices = 0

        self.h_start = 0
        self.h_end = 0

        self.v_start = 0
        self.v_end = 0
        
        self.frame_len = 0
        self.image_in_buffer = 0

class ImageMetadataProcessor:
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
        pass

    def process_metadata(self, buffer_idx, buffer):

        metadata = ImageMetadataContainer()

        metadata.buffer_idx = buffer_idx

        metadata.n_h = num_words(buffer[buffer_idx + WORD_H_OFFSET])
        metadata.timing_h = timing_words_not_found(buffer[buffer_idx + WORD_H_OFFSET])
        metadata.mismatch_h = timing_word_mismatch(buffer[buffer_idx + WORD_H_OFFSET])
        metadata.fifo_h = fifo_empty(buffer[buffer_idx + WORD_H_OFFSET])
        metadata.overload_h = overload_timing_words_exist(buffer[buffer_idx + WORD_H_OFFSET])

        metadata.n_v = num_words(buffer[buffer_idx + WORD_V_OFFSET])
        metadata.timing_v = timing_words_not_found(buffer[buffer_idx + WORD_V_OFFSET])
        metadata.mismatch_v = timing_word_mismatch(buffer[buffer_idx + WORD_V_OFFSET])
        metadata.fifo_v = fifo_empty(buffer[buffer_idx + WORD_V_OFFSET])
        metadata.overload_v = overload_timing_words_exist(buffer[buffer_idx + WORD_V_OFFSET])

        metadata.particle_count = buffer[buffer_idx + WORD_PC_OFFSET]
        metadata.num_slices = buffer[buffer_idx + WORD_NUM_SLICE_OFFSET]

        metadata.h_start = metadata.buffer_idx + METADATA_LENGTH
        metadata.h_end = metadata.h_start + metadata.n_h

        metadata.v_start = metadata.buffer_idx + METADATA_LENGTH + metadata.n_h
        metadata.v_end = metadata.v_start + metadata.n_v

        metadata.frame_len = METADATA_LENGTH + metadata.n_h + metadata.n_v
        metadata.image_in_buffer = (metadata.buffer_idx + metadata.frame_len) < 2048

        return metadata

@jit(nopython = True)
def num_words(word):
    # Bit masking out of a 16-bit number
    # to only get the 12 bit component
    return word & 0b0000111111111111

@jit(nopython = True)
def timing_words_not_found(word):
    # Bitmask to get 12th bit only, then bit shift right
    # 12 spots to keep only that bit
    return (word & 0b0001000000000000) >> 12

@jit(nopython = True)
def timing_word_mismatch(word):
    # Bitmask to get 13th bit only, then bit shift right
    # 13 spots to keep only that bit
    return (word & 0b0010000000000000) >> 13

@jit(nopython = True)
def fifo_empty(word):
    # Bitmask to get 14th bit only, then bit shift right
    # 14 spots to keep only that bit
    return (word & 0b0100000000000000) >> 14

@jit(nopython = True)
def overload_timing_words_exist(word):
    # Bitmask to get 15th bit only, then bit shift right
    # 15 spots to keep only that bit
    return (word & 0b1000000000000000) >> 15

class RawImageContainer:

    def __init__(self) -> None:
        self.raw_image_h = np.array([], dtype=np.uint16)
        self.raw_image_v = np.array([], dtype=np.uint16)

class ImageTimewordContainer:

    def __init__(self) -> None:
        self.timeword_h_upper = 0
        self.timeword_h_lower = 0
        self.timeword_v_upper = 0
        self.timeword_v_lower = 0

class RawImageExtractor:

    def __init__(self) -> None:
        self.raw_image_container = RawImageContainer()
        self.image_timeword_container = ImageTimewordContainer()

    def extract_raw_images(self, metadata, buffer):
        self.raw_image_container = RawImageContainer()

        raw_image_h = buffer[metadata.h_start:metadata.h_end]
        raw_image_v = buffer[metadata.v_start:metadata.v_end]

        if metadata.timing_h == 0:
            raw_image_h = raw_image_h[:-2]

        if metadata.timing_v == 0:
            raw_image_v = raw_image_v[:-2]

        self.raw_image_container.raw_image_h = raw_image_h
        self.raw_image_container.raw_image_v = raw_image_v

        return self.raw_image_container

    def extract_image_timewords(self, metadata, buffer):
        self.image_timeword_container = ImageTimewordContainer()
    
        raw_image_h = buffer[metadata.h_start:metadata.h_end]
        raw_image_v = buffer[metadata.v_start:metadata.v_end]

        if (metadata.timing_h == 0) and (len(raw_image_h) >= 2):
            self.image_timeword_container.timeword_h_upper = raw_image_h[-2]
            self.image_timeword_container.timeword_h_lower = raw_image_h[-1]

        if (metadata.timing_v == 0) and (len(raw_image_v) >= 2):
            self.image_timeword_container.timeword_v_upper = raw_image_v[-2]
            self.image_timeword_container.timeword_v_lower = raw_image_v[-1]

        return self.image_timeword_container

class DecodedImageContainer:

    def __init__(self) -> None:
        self.decoded_image_h = np.empty(0, dtype = decoded_word_type)
        self.decoded_image_v = np.empty(0, dtype = decoded_word_type)

class RawImageDecoder:

    def __init__(self) -> None:
        pass

    def decode_dual_channel_images(self, raw_image_container):
        
        decoded_image_container = DecodedImageContainer()
        decoded_image_container.decoded_image_h = decode_image(raw_image_container.raw_image_h)
        decoded_image_container.decoded_image_v = decode_image(raw_image_container.raw_image_v)
        
        return decoded_image_container

@jit
def decode_image(encoded_image):
    decoded_image = np.zeros(len(encoded_image), dtype = decoded_word_type)

    for i, word in enumerate(encoded_image):
        if word == 0x7fff:
            decoded_image['is_image_slice'][i] = 1
            decoded_image['is_start_slice'][i] = 1
            decoded_image['num_clear'][i] = 128
            decoded_image['num_shaded'][i] = 0
        elif word == 0x4000:
            decoded_image['is_image_slice'][i] = 1
            decoded_image['is_start_slice'][i] = 1
            decoded_image['num_clear'][i] = 0
            decoded_image['num_shaded'][i] = 128
        else:
            decoded_image['is_image_slice'][i] = ((word & 2**15) >> 15) == 0
            decoded_image['is_start_slice'][i] = (word & 2**14) >> 14
            decoded_image['num_shaded'][i] = (word & 0b0011111110000000) >> 7
            decoded_image['num_clear'][i]  = (word & 0b0000000001111111)

    valid_image_words = decoded_image['is_image_slice'] == True

    return decoded_image[valid_image_words]

class DecompressedImageContainer:

    def __init__(self) -> None:
        self.decompressed_image_h = np.array([], np.uint8)
        self.decompressed_image_v = np.array([], np.uint8)

class DecodedImageDecompressor:

    def __init__(self) -> None:
        pass

    def decompress_image(self, decoded_image_container):

        decompressed_image_container = DecompressedImageContainer()

        decompressed_image_container.decompressed_image_h = self.decompress_single_channel_image(decoded_image_container.decoded_image_h)
        decompressed_image_container.decompressed_image_v = self.decompress_single_channel_image(decoded_image_container.decoded_image_v)

        return decompressed_image_container

    def decompress_single_channel_image(self, decoded_image):

        if len(decoded_image) == 0:
            return []
        else:
            return decompress_complete_image(decoded_image)

@jit(nopython = True)
def get_complete_image_slice_inds(start_slice_flags):
    image_slice_id = np.cumsum(start_slice_flags)

    image_slice_inds = []
    
    for i in np.unique(image_slice_id):
        image_slice_inds.append(
            np.ravel(
                np.argwhere(image_slice_id == i)
            )
        )
    
    return image_slice_inds

@jit(nopython = True)
def decompress_complete_image(decoded_image):
    image_slice_inds = get_complete_image_slice_inds(decoded_image['is_start_slice'])

    image_slices = [int(x) for x in range(0)]
    #breakpoint()
    for i, slice_collection in enumerate(image_slice_inds):
        image_slice = [int(x) for x in range(0)]

        for slice_idx in slice_collection:

            """
            Be sure to append clear, THEN shaded
            Order matters very much here
            """

            image_slice += [CLEAR_VAL]*decoded_image['num_clear'][slice_idx]
            image_slice += [SHADED_VAL]*decoded_image['num_shaded'][slice_idx]
        
        # Add some clear bits to any incomplete slices

        if len(image_slice) < 128:
            image_slice += [CLEAR_VAL] * (128 - len(image_slice))

        # NOTE -> This is a TEMPORARY fix to a very strange encountered problem. To be sorted out later.

        if len(image_slice) > 128:
            image_slice += [CLEAR_VAL] * int(128*np.ceil(len(image_slice)/128) - len(image_slice) )

        image_slices += image_slice

    return np.array(image_slices, dtype=np.uint8)

class AssembledImageRecordContainer:

    def __init__(self) -> None:

        # Critical variables for the SPIF format

        self.buffer_id = np.array([], np.int32)
        self.image_sec = np.array([], np.int32)
        self.image_ns = np.array([], np.int32)

        self.image_len = np.array([], np.int32)
        self.image = np.array([], dtype = np.uint8)

        # Auxiliaries not neccessary for the format

        self.num_words = np.array([], dtype = np.uint16)
        self.timing_flag = np.array([], dtype = np.uint16)
        self.mismatch_flag = np.array([], dtype = np.uint16)
        self.fifo_flag = np.array([], dtype = np.uint16)
        self.overload_flag = np.array([], dtype = np.uint16)

        self.particle_count = np.array([], dtype = np.uint16)
        self.num_slices = np.array([], dtype = np.uint16)

        self.timeword_upper = np.array([], dtype = np.uint16)
        self.timeword_lower = np.array([], dtype = np.uint16)

        self.tas = np.array([], dtype = np.float32)

class ImageRecordAssembler:

    def __init__(self) -> None:
        pass

    def assemble_images(
        self,
        buffer_id,
        buffer_sec,
        buffer_ns,
        metadata_containers,
        timeword_containers,
        decompressed_image_containers,
        housekeeping
    ):

        image_container_h = AssembledImageRecordContainer()
        image_container_v = AssembledImageRecordContainer()

        image_container_h, image_container_v = self.set_buffer_info(
            buffer_id = buffer_id, 
            buffer_sec = buffer_sec, 
            buffer_ns = buffer_ns,        
            num_images = len(metadata_containers),    
            image_container_h = image_container_h,
            image_container_v = image_container_v
        )

        image_container_h, image_container_v = self.set_image_tas(
            housekeeping = housekeeping,
            num_images = len(metadata_containers),
            image_container_h = image_container_h,
            image_container_v = image_container_v
        )

        if len(metadata_containers) > 0:

            image_container_h, image_container_v = self.set_images(
                image_container_h,
                image_container_v,
                decompressed_image_containers
            )

            image_container_h, image_container_v = self.set_auxiliaries(
                metadata_containers, 
                timeword_containers,
                image_container_h, 
                image_container_v
            )

        return image_container_h, image_container_v

    def set_buffer_info(self, buffer_id, buffer_sec, buffer_ns, num_images, image_container_h, image_container_v):

        image_container_h.buffer_id = np.zeros(num_images, dtype = np.int32) + buffer_id
        image_container_h.image_sec = np.zeros(num_images, dtype = np.int32) + buffer_sec
        image_container_h.image_ns = np.zeros(num_images, dtype = np.int32) + buffer_ns

        image_container_v.buffer_id = np.zeros(num_images, dtype = np.int32) + buffer_id
        image_container_v.image_sec = np.zeros(num_images, dtype = np.int32) + buffer_sec
        image_container_v.image_ns = np.zeros(num_images, dtype = np.int32) + buffer_ns

        return image_container_h, image_container_v

    def set_image_tas(self, housekeeping, num_images, image_container_h, image_container_v):

        housekeeping = housekeeping[ housekeeping['incomplete_packet'] == 0 ]

        if len(housekeeping) == 0:
            tas = np.zeros(num_images)*np.nan
        else:
            tas = np.zeros(num_images, dtype = np.float32) + housekeeping['tas'][-1]

        image_container_h.tas = tas
        image_container_v.tas = tas

        return image_container_h, image_container_v

    def set_images(
        self,
        image_container_h,
        image_container_v,
        decompressed_image_containers
    ):
        image_container_h.image_len = np.array(
            [len(x.decompressed_image_h) / 128 for x in decompressed_image_containers],
            dtype = np.int32
        )

        image_container_v.image_len = np.array(
            [len(x.decompressed_image_v) / 128 for x in decompressed_image_containers],
            dtype = np.int32
        )

        image_container_h.image = np.concatenate(
            [x.decompressed_image_h for x in decompressed_image_containers]
        )

        image_container_v.image = np.concatenate(
            [x.decompressed_image_v for x in decompressed_image_containers]
        )

        return image_container_h, image_container_v

    def set_auxiliaries(
        self, 
        metadata, 
        timewords,
        image_container_h, 
        image_container_v
    ):

        image_container_h.num_words      = np.array([x.n_h for x in metadata], dtype = np.uint16)
        image_container_h.timing_flag    = np.array([x.timing_h for x in metadata], dtype = np.uint16)
        image_container_h.mismatch_flag  = np.array([x.mismatch_h for x in metadata], dtype = np.uint16)
        image_container_h.fifo_flag      = np.array([x.fifo_h for x in metadata], dtype = np.uint16)
        image_container_h.overload_flag  = np.array([x.overload_h for x in metadata], dtype = np.uint16)
        image_container_h.particle_count = np.array([x.particle_count for x in metadata], dtype = np.uint16)
        image_container_h.num_slices     = np.array([x.num_slices for x in metadata], dtype = np.uint16)
        image_container_h.timeword_upper = np.array([x.timeword_h_upper for x in timewords], dtype = np.uint16)
        image_container_h.timeword_lower = np.array([x.timeword_h_lower for x in timewords], dtype = np.uint16)

        image_container_v.num_words      = np.array([x.n_v for x in metadata], dtype = np.uint16)
        image_container_v.timing_flag    = np.array([x.timing_v for x in metadata], dtype = np.uint16)
        image_container_v.mismatch_flag  = np.array([x.mismatch_v for x in metadata], dtype = np.uint16)
        image_container_v.fifo_flag      = np.array([x.fifo_v for x in metadata], dtype = np.uint16)
        image_container_v.overload_flag  = np.array([x.overload_v for x in metadata], dtype = np.uint16)
        image_container_v.particle_count = np.array([x.particle_count for x in metadata], dtype = np.uint16)
        image_container_v.num_slices     = np.array([x.num_slices for x in metadata], dtype = np.uint16)
        image_container_v.timeword_upper = np.array([x.timeword_v_upper for x in timewords], dtype = np.uint16)
        image_container_v.timeword_lower = np.array([x.timeword_v_lower for x in timewords], dtype = np.uint16)

        return image_container_h, image_container_v

# A utility method to define the extra image related variables in a netcdf file

def add_auxiliary_core_variables(spiffile, inst_name):
        coregrp = spiffile.rootgrp[inst_name]['core']

        spiffile.create_variable(
            coregrp,
            'num_words',
            'u2',
            ('Images',),
            {
                'long_name':'Number of data words present in image',
                'units':'counts'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'timing_flag',
            'u2',
            ('Images',),
            {
                'long_name':'Timing flag for image',
                'units':'boolean'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'mismatch_flag',
            'u2',
            ('Images',),
            {
                'long_name':'Mismatch flag for image',
                'units':'boolean'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'fifo_flag',
            'u2',
            ('Images',),
            {
                'long_name':'FIFO flag for image',
                'units':'boolean'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'overload_flag',
            'u2',
            ('Images',),
            {
                'long_name':'Overload flag for image',
                'units':'binary'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'particle_count',
            'u2',
            ('Images',),
            {
                'long_name':'Number of particles detected in image',
                'units':'counts'
            },
            chunksizes=(TIME_CHUNK,)
        )
        
        spiffile.create_variable(
            coregrp,
            'num_slices',
            'u2',
            ('Images',),
            {
                'long_name':'Number of slices detected in the image',
                'units':'counts'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'timeword_upper',
            'u2',
            ('Images',),
            {
                'long_name':'Upper 16 bits of timeword for image',
                'units':'clock ticks'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'timeword_lower',
            'u2',
            ('Images',),
            {
                'long_name':'Lower 16 bits of timeword for image',
                'units':'clock ticks'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'tas',
            'f',
            ('Images',),
            {
                'long_name':'True airspeed as recorded by probe',
                'units':'m/s'
            },
            chunksizes=(TIME_CHUNK,)
        )