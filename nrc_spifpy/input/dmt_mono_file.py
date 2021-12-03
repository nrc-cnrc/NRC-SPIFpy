#!/usr/bin/env python
# coding: utf-8

import datetime
import math

import numpy

from nrc_spifpy.input.dmt_binary_file import DMTBinaryFile
from nrc_spifpy.images import Images
from nrc_spifpy.utils import bin_to_int, convert_uint8_to_bitlist


class DMTMonoFile(DMTBinaryFile):
    """ Class representing monoscale binary format for DMT instruments (CIP, PIP).
    Implements methods specific to decompressing and processing images from
    this type of binary file.

    Class Attributes
    ----------------
    syncword : numpy array
        Numpy array of syncword used in DMT monoscale format.

    Parameters
    ----------
    filename : str
        Filename of current file.
    inst_name : str
        Name of current instrument, e.g., CIP, PIP.
    """
    syncword = numpy.array([170] * 8, dtype='B')

    def __init__(self, filename, inst_name, resolution):
        super().__init__(filename, inst_name, resolution)
        self.diodes = 64
        self.file_dtype = numpy.dtype([('year', 'u2'),
                                       ('month', 'u2'),
                                       ('day', 'u2'),
                                       ('hour', 'u2'),
                                       ('minute', 'u2'),
                                       ('second', 'u2'),
                                       ('ms', 'u2'),
                                       ('weekday', 'u2'),
                                       ('data', '(4096, )B')
                                       ])

        self.aux_channels = ['image_count', 'dof_flag']

    def read(self):
        dummy = 0
        try:
            super().read()
        except ValueError:
            err = True
            while dummy < 32 and err:
                dummy += 1
                self.file_dtype = numpy.dtype([('year', 'u2'),
                                               ('month', 'u2'),
                                               ('day', 'u2'),
                                               ('hour', 'u2'),
                                               ('minute', 'u2'),
                                               ('second', 'u2'),
                                               ('ms', 'u2'),
                                               ('weekday', 'u2'),
                                               ('dummy', f'({dummy}, )B'),
                                               ('data', '(4096, )B')
                                               ])
                try:
                    super().read()
                    err = False
                except ValueError as v:
                    pass

    def process_frame(self, frame):
        """ Method to process single frame of image data. Decompresses image
        data from frame and passes resulting image data to extract_images
        method to extract image info from image buffer.

        Parameters
        ----------
        frame : int
            Frame number in self.data to process.

        Returns
        -------
        Images object
            Images object containing image info in current frame.
        """
        data = self.data[frame]
        record = data['data']
        i = 0
        frame_decomp = []
        while i < 4096:
            b1 = record[i]
            counts = (b1 & 31) + 1
            if b1 & 128 == 128:
                frame_decomp.extend([00] * counts)
            elif b1 & 64 == 64:
                frame_decomp.extend([255] * counts)
            elif b1 & 32 == 32:
                i = i + 1
                continue
            else:
                frame_decomp.extend(record[i + 1:i + counts + 1])
                i += counts
            i += 1
        date = datetime.datetime(data['year'],
                                 data['month'],
                                 data['day'])
        # if frame % 1000 == 0:
        #     print('At frame ' + str(frame) + ' of ' + str(len(self.data)))
        return self.extract_images(frame, frame_decomp, date)

    def extract_images(self, frame, frame_decomp, date):
        """ Extracts images and associated info from decompressed image
        buffer. The image image, length, timestamp in sec and ns, image
        count, dof flag, and buffer index are extracted for each image.

        Parameters
        ----------
        frame : int
            Frame number of current frame being processed.
        frame_decomp : array
            Array of decompressed image buffer.
        data : datetime object
            Date of current image buffer.

        Returns
        -------
        Images object
            Images object containing image data extracted from image
            buffer.
        """
        images = Images(self.aux_channels)

        ii = 0
        old_ii = 0
        while ii < len(frame_decomp) - 16:
            ii = self.search_syncword(frame_decomp, ii)
            ii_next = self.search_syncword(frame_decomp, ii + 8)
            # print(ii - old_ii)
            frameval = frame_decomp[ii: ii + 8]
            if ii >= len(frame_decomp) - 17:
                break
            header_bin = convert_uint8_to_bitlist(frame_decomp[ii + 8: ii + 16])

            image_count = bin_to_int(header_bin[0:16])
            hour = bin_to_int(header_bin[51:56])
            minute = bin_to_int(header_bin[45:51])
            second = bin_to_int(header_bin[39:45])
            millisecond = bin_to_int(header_bin[29:39])
            nanosecond_eigths = bin_to_int(header_bin[16:29])
            slice_count_old = bin_to_int(header_bin[57:64]) - 1
            slice_count = int((ii_next - ii) / 8) - 2
            # print(slice_count, slice_count_old)
            dof_flag = bin_to_int(header_bin[56:57])
            im = frame_decomp[ii + 16: ii + 16 + (slice_count * 8)]
            image = convert_uint8_to_bitlist(im)
            ii += 16 + (slice_count) * 8
            # old_ii = ii

            image_time = date + datetime.timedelta(hours=hour,
                                                   minutes=minute,
                                                   seconds=second)

            epoch_time = (image_time - self.start_date).total_seconds()
            images.ns.append(millisecond * 1e6 + nanosecond_eigths / 125)
            images.sec.append(epoch_time)
            if len(image) % self.diodes != 0:
                pad_amount = math.ceil(len(image) / self.diodes) * self.diodes - len(image)
                image = numpy.append(image, numpy.ones(pad_amount))
            images.length.append(len(image) // self.diodes)
            image = numpy.array(image)
            # image = numpy.reshape(image, (-1, self.diodes))
            images.image.append(image)
            images.image_count.append(image_count)
            images.buffer_index.append(frame)
            images.dof_flag.append(dof_flag)

        return images

    def search_syncword(self, frame_decomp, i):
        """ Searches image buffer for next image using syncword. If not found
        returns index of end of current image buffer.

        Parameters
        ----------
        frame_decomp : array
            Array of decompressed image buffer.
        i : int
            Index in frame_decomp to search from.

        Returns
        -------
        int
            Index value of start of next syncword.
        """
        try:
            while True:
                i = frame_decomp.index(self.syncword[0], i)
                if (frame_decomp[i: i + 8] == self.syncword).all():
                    return i
                i += 1
        except (ValueError, AttributeError, IndexError):
            return len(frame_decomp)
