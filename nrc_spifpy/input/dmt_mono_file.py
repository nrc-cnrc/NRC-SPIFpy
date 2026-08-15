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
    AUX_DTYPES = {
        'image_count': 'u2',
        'dof_flag': 'u1',
    }
    AUX_ATTRS = {
        'image_count': {
            'long_name': 'DMT particle image counter',
            'units': '1',
            'comment': (
                '16-bit counter stored modulo 2^16; discontinuities can '
                'indicate dropped particles.'
            ),
        },
        'dof_flag': {
            '_FillValue': numpy.uint8(255),
            'long_name': 'DMT depth-of-field flag',
            'units': '1',
            'flag_values': numpy.array([0, 1], dtype=numpy.uint8),
            'flag_meanings': (
                'out_of_focus meets_depth_of_field_requirement'
            ),
            'description': (
                'Active-high DMT probe flag: 1 indicates that the particle '
                'meets the depth-of-field requirement for sizing; 0 indicates '
                'rejection. Missing flags are stored as the fill value.'
            ),
        },
    }

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
    
    '''
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

        History
        -------
        2024-08-10: Yongjie Huang (OU, huangynj@gmail.com), fixed the issue when particle spanning two frames.

        """
        data = self.data[frame]
        record = data['data']
        try:
            record_next = self.data[frame+1]['data']
            next_record_exists = True
            record = numpy.concatenate((record, record_next))
        except Exception as e:
            # raise e
            # pass
            next_record_exists = False

        i = 0
        frame_decomp = []
        # while i < 4096:
        while True:
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
                # if len(record[i + 1:i + counts + 1]) != counts:
                #     print(frame, i, len(record[i + 1:i + counts + 1]), counts)
                #     print(frame, frame_decomp)
                i += counts
            i += 1

            # Note: The condition "i > 4096 + 8" ensures that we read the first full sequence of "7, [170] * 8"
            # from the next frame. Or, the first partical of the next frame will be missed.
            if next_record_exists and (len(frame_decomp) > 16) and \
               (frame_decomp[-8:] == self.syncword).all() and (i > 4096 + 8): 
                break  
            if (not next_record_exists) and (i >= 4096):
                break

        date = datetime.datetime(data['year'],
                                 data['month'],
                                 data['day'])
        # if frame % 1000 == 0:
        #     print('At frame ' + str(frame) + ' of ' + str(len(self.data)))
        # if 98 <= frame <= 99:
        #     print('i = ', i, frame, frame_decomp, date, '\n', frame_decomp[-8:], '\n')
        return self.extract_images(frame, frame_decomp, date)
    '''

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

        History
        -------
        2024-08-14: Yongjie Huang (OU, huangynj@gmail.com), updated the function with a robust method if compressed image buffers don't begin with a RLEHB:
                        1) Search syncword indices before decompressing frame
                        2) Decompress data only between syncwords
        2024-08-10: Yongjie Huang (OU, huangynj@gmail.com), fixed the issue when particle spanning two frames.

        """
        data = self.data[frame]
        record = data['data']
        try:
            record_next = self.data[frame+1]['data']
            next_record_exists = True
            record = numpy.concatenate((record, record_next))
        except Exception as e:
            # raise e
            # pass
            next_record_exists = False

        #-- Search syncword indices before decompressing frame
        record_list = record.tolist()
        syncword_idx = []
        ii = 0
        while True:
            ii = self.search_syncword(record_list, ii)

            if ii >= len(record_list) - 17:
                break

            if ii > 0:                             # Ensure a RLEHB before the syncword
                syncword_idx.append(ii)

            if next_record_exists and (ii > 4096): # Has reached the first full sequence of "RLEHB, [170] * 8" from the next frame
                break

            ii = ii + 8

        #-- decompress data only between syncwords
        frame_decomp = []
        for i_sw in range(len(syncword_idx)-1):
            n_170 = (record[syncword_idx[i_sw]-1] & 31) + 1 # Decompress the RLEHB before the syncword, because it is not always "7" !!!
            frame_decomp.extend([170] * n_170)

            ii_s = syncword_idx[i_sw] + n_170
            ii_e = syncword_idx[i_sw+1] - 1                 # Exclude the RLEHB (e.g., "7") before the syncword

            i = ii_s
            while i < ii_e:
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

        frame_decomp.extend(self.syncword) # Add the final syncword

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
            images.ns.append(millisecond * 1e6 + nanosecond_eigths * 125) # Fix nanosecond, Yongjie Huang, 2024-08-09.
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
