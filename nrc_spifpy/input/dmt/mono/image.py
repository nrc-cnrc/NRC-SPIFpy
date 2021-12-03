import numpy as np

from numba import njit

from math import ceil

from tqdm import tqdm

from nrc_spifpy.spif import TIME_CHUNK

class ImageMetadataContainer:

    def __init__(self) -> None:
        self.hours = 0
        self.minutes = 0
        self.seconds = 0
        self.milliseconds = 0
        self.nanoseconds = 0

        self.particle_count = 0
        self.slice_count = 0
        self.dof_flag = 0

class ImageMetadataProcessor:

    def __init__(self) -> None:
        pass

    def process_image_metadata(self, image):
        timestamp_bitlist = get_timestamp_bitlist(image)

        metadata_container = ImageMetadataContainer()

        metadata_container.hours        = get_hours(timestamp_bitlist)
        metadata_container.minutes      = get_minutes(timestamp_bitlist)
        metadata_container.seconds      = get_seconds(timestamp_bitlist)
        metadata_container.milliseconds = get_milliseconds(timestamp_bitlist)
        metadata_container.nanoseconds  = get_nanoseconds(timestamp_bitlist)

        metadata_container.particle_count = get_particle_count(image)
        metadata_container.slice_count    = get_slice_count(image)
        metadata_container.dof_flag       = get_dof(image)

        return metadata_container

def get_timestamp_bitlist(image):
    timestamp_bytes = image[6:1:-1]

    timestamp_bitlist = int_to_bin(timestamp_bytes, big_endian = True)

    return timestamp_bitlist

def get_hours(timestamp_bitlist):
    hours = bin_to_int(timestamp_bitlist[0:5], big_endian = True)
    return hours

def get_minutes(timestamp_bitlist):
    minutes = bin_to_int(timestamp_bitlist[5:11], big_endian = True)
    return minutes

def get_seconds(timestamp_bitlist):
    seconds = bin_to_int(timestamp_bitlist[11:17], big_endian = True)
    return seconds

def get_milliseconds(timestamp_bitlist):
    milliseconds = bin_to_int(timestamp_bitlist[17:27], big_endian = True)
    return milliseconds

def get_nanoseconds(timestamp_bitlist):
    nanoseconds = bin_to_int(timestamp_bitlist[27:40], big_endian = True) * 125
    return nanoseconds

@njit
def get_particle_count(image):
    particle_count = image[:2].view(np.dtype('<u2'))[0]
    return particle_count

@njit
def get_slice_count(image_data):
    return(image_data[7] & 0b11111110) >> 1

@njit
def get_dof(image_data):
    return(image_data[7] & 0b00000001)


class ImageDataContainer:

    def __init__(self) -> None:
        self.image = np.array([], dtype = np.dtype('B'))
        self.image_len = 0

class ImageDataProcessor:

    def __init__(self) -> None:
        pass

    def process_image_data(self, decompressed_image):


        image_container = ImageDataContainer()

        image = decompressed_image[8:] # skipping the header

        image_len = len(image)

        if image_len % 8 != 0:
            padded_image_len = int( ceil(image_len/8)*8 )
            padding = np.zeros(padded_image_len - image_len, dtype = np.dtype('B'))
            image = np.append(image, padding)

        image = np.reshape(image, (-1, 8))
        image = np.fliplr(image)
        image = np.ravel(image)

        image_bitlist = int_to_bin(image)

        image_container.image = image_bitlist
        image_container.image_len = int(len(image_bitlist) / 64)

        return image_container

class AssembledImageContainer:

    def __init__(self) -> None:
        self.image_sec = np.array([], dtype = np.dtype('u4'))
        self.image_ns = np.array([], dtype = np.dtype('u4'))
        self.buffer_index = np.array([], dtype = np.dtype('u4'))
        self.buffer_sec = np.array([], dtype = np.dtype('u4'))
        self.buffer_ns = np.array([], dtype = np.dtype('u4'))
        self.particle_count = np.array([], dtype = np.dtype('u2'))
        self.slice_count = np.array([], dtype = np.dtype('B'))
        self.dof_flag = np.array([], dtype = np.dtype('B'))
        self.image_len = np.array([], dtype = np.dtype('u2'))
        self.image = np.array([], dtype = np.dtype('B'))

class ImageAssembler:

    def __init__(self) -> None:
        pass

    def assemble_images(self, metadata_containers, image_containers):
        assembled_image_container = AssembledImageContainer()

        assembled_image_container = self.set_buffer_info(
            assembled_image_container,
            num_images = len(metadata_containers)
        )

        assembled_image_container = self.set_image_sec(assembled_image_container, metadata_containers)
        assembled_image_container = self.set_image_ns(assembled_image_container, metadata_containers)
        assembled_image_container = self.set_particle_count(assembled_image_container, metadata_containers)
        assembled_image_container = self.set_slice_count(assembled_image_container, metadata_containers)
        assembled_image_container = self.set_dof_flag(assembled_image_container, metadata_containers)

        assembled_image_container = self.set_image_len(assembled_image_container, image_containers)
        assembled_image_container = self.set_image(assembled_image_container, image_containers)

        return assembled_image_container

    def set_buffer_info(self, assembled_record_container, num_images):
        assembled_record_container.buffer_index = np.zeros(num_images, dtype = np.dtype('u4'))
        assembled_record_container.buffer_sec = np.zeros(num_images, dtype = np.dtype('u4'))
        assembled_record_container.buffer_ns = np.zeros(num_images, dtype = np.dtype('u4'))

        return assembled_record_container

    def set_image_sec(self, assembled_record_container, metadata_containers):

        assembled_record_container.image_sec = np.zeros(len(metadata_containers), dtype=np.dtype('u4'))

        for i in range(len(metadata_containers)):

            assembled_record_container.image_sec[i] = \
              metadata_containers[i].hours*3600.0 \
            + metadata_containers[i].minutes*60 \
            + metadata_containers[i].seconds

        return assembled_record_container

    def set_image_ns(self, assembled_record_container, metadata_containers):

        assembled_record_container.image_ns = np.zeros(len(metadata_containers), dtype=np.dtype('u4'))

        for i in range(len(metadata_containers)):
            
            assembled_record_container.image_ns[i] = \
              metadata_containers[i].milliseconds*1e6 \
            + metadata_containers[i].nanoseconds

        return assembled_record_container

    def set_particle_count(self, assembled_record_container, metadata_containers):

        assembled_record_container.particle_count = np.zeros(len(metadata_containers), dtype=np.dtype('u2'))

        for i in range(len(metadata_containers)):
            assembled_record_container.particle_count[i] = metadata_containers[i].particle_count

        return assembled_record_container

    def set_slice_count(self, assembled_record_container, metadata_containers):

        assembled_record_container.slice_count = np.zeros(len(metadata_containers), dtype=np.dtype('B'))

        for i in range(len(metadata_containers)):
            assembled_record_container.slice_count[i] = metadata_containers[i].slice_count

        return assembled_record_container

    def set_dof_flag(self, assembled_record_container, metadata_containers):

        assembled_record_container.dof_flag = np.zeros(len(metadata_containers), dtype=np.dtype('B'))

        for i in range(len(metadata_containers)):
            assembled_record_container.dof_flag[i] = metadata_containers[i].dof_flag

        return assembled_record_container

    def set_image_len(self, assembled_record_container, image_containers):

        assembled_record_container.image_len = np.zeros(len(image_containers), dtype=np.dtype('u2'))

        for i in range(len(image_containers)):
            assembled_record_container.image_len[i] = image_containers[i].image_len

        return assembled_record_container

    def set_image(self, assembled_record_container, image_containers):

        assembled_record_container.image = np.zeros([], dtype=np.dtype('B'))

        image_arr = [x.image for x in image_containers]

        assembled_record_container.image = np.concatenate(image_arr)

        return assembled_record_container

def get_image_boundaries(data, syncword_sequence_inds):
    image_bound_start = syncword_sequence_inds
    image_bound_end = np.zeros_like(syncword_sequence_inds)
    image_bound_end[0:-1] = syncword_sequence_inds[1:]
    image_bound_end[-1] = len(data)
    
    image_bounds = np.vstack([image_bound_start, image_bound_end])
    image_bounds = image_bounds.T

    return image_bounds

def find_syncword_sequences(data):

    syncword = 0xAA
    conv_kernel = np.array([1,1,1,1,1,1,1,1], dtype = 'B')
    syncword_inds = (data == syncword)
    
    """
    'valid' makes the convolution start inside the array like this:
    |0,1,2|
    |0,1,2,3,4,5....|
    
    instead of this
    
    |0,1,2|
          |0,1,2,3,4,5...|
    """
    
    syncword_inds_conv = np.convolve(syncword_inds, conv_kernel, 'valid')
    
    syncword_sequence_inds = np.argwhere(syncword_inds_conv == 8)
    syncword_sequence_inds = np.ravel(syncword_sequence_inds)
    
    return syncword_sequence_inds

@njit
def int_to_bin(arr, big_endian = True, flat = True):
    val_bitstream = np.zeros((len(arr),8), dtype = np.dtype('B'))

    for i in range(len(arr)):
        for j in range(8):
            val_bitstream[i,j] = (arr[i] >> j) & 0b00000001
            
    if big_endian:
        val_bitstream = np.fliplr(val_bitstream)

    val_bitstream_flat = np.ravel(val_bitstream)

    return val_bitstream_flat

@njit
def bin_to_int(bin_list, big_endian=True):

    if big_endian:
        step = 1
    else:
        step = -1

    bin_list = bin_list[::step]
    bin_int = 0
    for bit in bin_list:
        bin_int = (bin_int << 1) | bit
    return int(bin_int)

def add_auxiliary_core_variables(spiffile, inst_name):
        coregrp = spiffile.rootgrp[inst_name]['core']

        spiffile.create_variable(
            coregrp,
            'dof_flag',
            'B',
            ('Images',),
            {
                'long_name':'Depth of Field Flag',
                'units':'boolean'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'particle_count',
            'u2',
            ('Images',),
            {
                'long_name':'Total counts of all particles',
                'units':'Counts'
            },
            chunksizes=(TIME_CHUNK,)
        )

        spiffile.create_variable(
            coregrp,
            'slice_count',
            'B',
            ('Images',),
            {
                'long_name':'Number of slices for this image',
                'units':'None'
            },
            chunksizes=(TIME_CHUNK,)
        )