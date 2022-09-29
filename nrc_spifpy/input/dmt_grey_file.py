#!/usr/bin/env python
# coding: utf-8

import datetime

import numpy

from . import DMTBinaryFile
from ..images import Images
from ..utils import bin_to_int


class DMTGreyFile(DMTBinaryFile):
    """ Class representing greyscale binary format for DMT instruments (CIPGS).
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
    resolution : int
        Resolution in microns of probe
    """
    syncword = numpy.array([3] * 128)

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
        self.aux_channels = ['tas', 'image_count']

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
        pairs = [0, 0, 0, 0]
        while i < 4096:
            b1 = record[i]

            if b1 & 128 == 128:
                counts = b1 & 127
                frame_decomp.extend([pairs[3]] * counts)
            else:
                pairs = [int((b1 & 64) / 64),
                         int((b1 & 48) / 16),
                         int((b1 & 12) / 4),
                         int(b1 & 3)]
                if pairs[0] == 1:
                    frame_decomp.extend(pairs[1:])
                elif pairs[1] == 1:
                    frame_decomp.extend(pairs[2:])
                elif pairs[2] == 1:
                    frame_decomp.append(pairs[3])
            i += 1

        date = datetime.datetime(data['year'],
                                 data['month'],
                                 data['day'])
        # if frame % 1000 == 0:
        #     print('At frame ' + str(frame) + ' of ' + str(len(self.data)))

        return self.extract_images(frame, frame_decomp, date)

    def extract_images(self, frame, frame_decomp, date, mono=False):
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

        ii = self.search_syncword(frame_decomp, ii)
        while ii < len(frame_decomp) - 64:
            header = frame_decomp[ii: ii + 64]
            header8 = []
            for i in range(0, 64, 4):
                elem = 0
                for j in range(4):
                    h = header[i + j]
                    h = int((h & 2) / 2 + (h & 1) * 2)
                    elem += 2 ** (2 * (3 - j)) * h
                header8.append(elem)

            header8 = numpy.array(header8, dtype=numpy.uint8)
            header_bin = numpy.unpackbits(header8)

            slice_count = bin_to_int(header_bin[120:128])
            hour = bin_to_int(header_bin[115:120])
            minute = bin_to_int(header_bin[109:115])
            second = bin_to_int(header_bin[103:109])
            millisecond = bin_to_int(header_bin[93:103])
            microsecond = bin_to_int(header_bin[83:93])
            nanosecond_eigths = bin_to_int(header_bin[80:83])
            image_count = bin_to_int(header_bin[64:80])
            tas = bin_to_int(header_bin[56:64])
            image = frame_decomp[ii + 64: ii + 64 + (slice_count * 64)]
            if mono:
                for i, val in enumerate(image):
                    image[i] -= 2
            ii = ii + 64 + ((slice_count - 1) * 64)
            if ii < len(frame_decomp) - 128:
                ii = self.search_syncword(frame_decomp, ii)
            else:
                break

            image_time = date + datetime.timedelta(hours=int(hour),
                                                   minutes=int(minute),
                                                   seconds=int(second))

            epoch_time = (image_time - self.start_date).total_seconds()
            images.ns.append(millisecond * 1000 +
                             microsecond * 1e6 +
                             nanosecond_eigths / 125)
            images.sec.append(epoch_time)
            images.length.append(len(image) // self.diodes)
            image = numpy.array(image)
            # image = numpy.reshape(image, (-1, self.diodes))
            images.image.append(image)
            images.buffer_index.append(frame)
            images.tas.append(tas)
            images.image_count.append(image_count)

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
                if (frame_decomp[i: i + 128] == self.syncword).all():
                    break
                i += 1

            while frame_decomp[i] == 3:
                i += 1
            return i
        except (ValueError, AttributeError, IndexError):
            return len(frame_decomp)
