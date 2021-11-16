import datetime

import numpy as np

from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.images import Images
from nrc_spifpy.input.spec.buffer import Buffer

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
            #self.write_data(spiffile, buffer.image_containers['h'], buffer.image_containers['v'])

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