import datetime

import numpy as np

from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.images import Images
from nrc_spifpy.input.spec.buffer import Buffer
from nrc_spifpy.input.spec.image import AssembledImageRecordContainer
from nrc_spifpy.input.spec.image import add_auxiliary_core_variables

from tqdm import tqdm

import joblib as jb

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
        self.chunk_size = 250


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

        group_no_channel = self.name
        group_h_channel = self.name + '-H'
        group_v_channel = self.name + '-V'

        if self.name == '2DS':
            spiffile.create_inst_group(group_h_channel)
            spiffile.create_inst_group(group_v_channel)
            spiffile.set_filenames_attr(group_h_channel, self.filename)
            spiffile.set_filenames_attr(group_v_channel, self.filename)

            add_auxiliary_core_variables(spiffile, group_h_channel)
            add_auxiliary_core_variables(spiffile, group_v_channel)
        else:
            spiffile.create_inst_group(group_no_channel)
            spiffile.set_filenames_attr(group_no_channel, self.filename)

            add_auxiliary_core_variables(self, spiffile, group_no_channel)

        spiffile.write_buffer_info(self.start_date, self.datetimes)

        pbar1 = tqdm(desc='Processing frames',
                     total=len(self.data),
                     unit='frame')

        chunk_intervals = self.get_chunk_intervals()

        for i in range(len(chunk_intervals) - 1):
            chunk_slice = slice(chunk_intervals[i], chunk_intervals[i + 1])
            
            buffer_ids = [x for x in range(chunk_intervals[i], chunk_intervals[i + 1])]
            buffer_timestamps = self.timestamps[chunk_slice]
            buffers = self.data[chunk_slice]['data']

            chunk_processor = ChunkedBufferProcessor(buffer_ids, buffer_timestamps, buffers)
            chunk_processor.process_buffers()
            images_h, images_v = chunk_processor.merge_assembled_images()

            if self.name == '2DS':
                self.write_data(spiffile, group_h_channel, images_h)
                self.write_data(spiffile, group_v_channel, images_v)
            else:
                self.write_data(spiffile, group_no_channel, images_h)

            pbar1.update(chunk_intervals[i + 1] - chunk_intervals[i])

    def get_chunk_intervals(self):
        data_len = len(self.data)

        chunk_intervals = np.arange(0, data_len, self.chunk_size)

        if chunk_intervals[-1] != data_len:
            chunk_intervals[-1] = data_len

        return chunk_intervals


    def write_data(self, spiffile, instrument, images):
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

        instgrp = spiffile.rootgrp[instrument]
        coregrp = spiffile.rootgrp[instrument]['core']

        dim_size = len(instgrp.dimensions['Images'])
        px_size = len(instgrp.dimensions['Pixels'])

        # Primary, critical SPIF variables

        spiffile.write_variable(coregrp, 'image_sec', images.image_sec, dim_size)
        spiffile.write_variable(coregrp, 'image_ns', images.image_ns, dim_size)
        spiffile.write_variable(coregrp, 'image_len', images.image_len, dim_size)
        spiffile.write_variable(coregrp, 'buffer_index', images.buffer_id, dim_size)
        spiffile.write_variable(coregrp, 'image', images.image, px_size)

        # Auxiliary SPIF variables

        spiffile.write_variable(coregrp, 'num_words', images.num_words, dim_size)
        spiffile.write_variable(coregrp, 'timing_flag', images.timing_flag, dim_size)
        spiffile.write_variable(coregrp, 'mismatch_flag', images.mismatch_flag, dim_size)
        spiffile.write_variable(coregrp, 'fifo_flag', images.fifo_flag, dim_size)
        spiffile.write_variable(coregrp, 'overload_flag', images.overload_flag, dim_size)
        spiffile.write_variable(coregrp, 'particle_count', images.particle_count, dim_size)
        spiffile.write_variable(coregrp, 'num_slices', images.num_slices, dim_size)
        spiffile.write_variable(coregrp, 'timeword_upper', images.timeword_upper, dim_size)
        spiffile.write_variable(coregrp, 'timeword_lower', images.timeword_lower, dim_size)
        spiffile.write_variable(coregrp, 'tas', images.tas, dim_size)

        spiffile.rootgrp.sync()

class ChunkedBufferProcessor:

    def __init__(self, buffer_ids, buffer_timestamps, buffers):
        self.buffer_ids = buffer_ids
        self.buffer_timestamps = buffer_timestamps
        self.buffers = buffers

        self.output = [None]*len(buffers)


    def process_buffers(self):

        for i in range(len(self.output)):
            self.output[i] = self.process_buffer(
                self.buffer_ids[i],
                self.buffer_timestamps[i],
                self.buffers[i]
            )

    def process_buffer(self, buffer_id, buffer_timestamp, buffer):

        buffer = Buffer(buffer_id, buffer_timestamp, buffer)
        return buffer.assembled_images

    def merge_assembled_images(self):

        image_container_h = AssembledImageRecordContainer()
        image_container_v = AssembledImageRecordContainer()

        for k in image_container_h.__dict__.keys():
            image_container_h.__dict__[k] = np.concatenate([x['h'].__dict__[k] for x in self.output])
            image_container_v.__dict__[k] = np.concatenate([x['v'].__dict__[k] for x in self.output])

        return image_container_h, image_container_v

