#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import datetime
import math
import os
import time

import numpy
from tqdm import tqdm

from . import BinaryFile
from .input_utils import read_sea_image_data, read_sea_tas
from ..images import Images
from ..utils import convert_uint32_to_bitlist

MAX_PROCESSORS = 20


class TwoDFile(BinaryFile):

    syncword = int('0x55000000', 16)

    def __init__(self, filename, inst_name, resolution):
        super().__init__(filename, inst_name, resolution)
        self.diodes = 32
        self.file_dtype = numpy.dtype([('year', 'u2'),
                                       ('month', 'u2'),
                                       ('day', 'u2'),
                                       ('hour', 'u2'),
                                       ('minute', 'u2'),
                                       ('second', 'u2'),
                                       ('ms', 'u2'),
                                       ('weekday', 'u2'),
                                       ('data', '(1024, )u4')
                                       ])
        self.aux_channels = ['tas', 'timeword']

    def read(self):

        self.tas = read_sea_tas(self.filename, '2dc', self.resolution)
        self.data = read_sea_image_data(self.filename, self.file_dtype, '2d', self.name)

        self.get_start_date()
        self.calc_buffer_datetimes()

    def process_file(self, spiffile, processors=None):
        """ Method to process file using multiple processors. Calls
        process_image method implemented in children classes. Passes image
        data to spiffile object to be written to file.

        Parameters
        ----------
        spiffile : SPIFFile object
            SPIFFile object of current SPIF NetCDF output file
        processors : int, optional
            Number of processors to use for parallel image processing. If none
            is provided, defaults to number of processors minus one.
        """

        if processors is None:
            processors = os.cpu_count()
            if os.cpu_count() > 1:
                processors -= 1

        if processors > MAX_PROCESSORS:
            processors = MAX_PROCESSORS

        spiffile.set_start_date(self.start_date.strftime('%Y-%m-%d %H:%M:%S %z'))

        process_until = len(self.data)
        data_chunk = range(0, process_until)

        tot_images = 0

        spiffile.create_inst_group(self.name)
        spiffile.write_buffer_info(self.start_date, self.datetimes)
        spiffile.set_filenames_attr(self.name, self.filename)

        pbar1 = tqdm(desc='Processing frames',
                     total=process_until,
                     unit='frame')
        pbar2 = tqdm(desc='Writing frames',
                     total=process_until,
                     unit='frame')
        t00 = time.time()

        i = 0
        chunksize = 500
        max_write_queue = 8
        images_remaining = True
        futures = []
        with ProcessPoolExecutor(max_workers=processors) as executor:
            while True:
                while len(futures) <= max_write_queue and images_remaining:
                    futures.append(executor.submit(self.process_frames,
                                                   data_chunk[i: i + chunksize]))
                    i += chunksize
                    if i >= process_until:
                        images_remaining = False
                    pbar1.update(chunksize)

                done, running = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    indx = futures.index(f)
                    if indx == 0:
                        pbar2.update(chunksize)
                        images = f.result()
                        tot_images += len(images)
                        t0 = time.time()
                        if len(images) > 0:
                            images.conv_to_array(self.diodes)
                            spiffile.write_images(self.name, images)

                        futures.pop(indx)

                if not images_remaining and len(futures) == 0:
                    break

        pbar1.close()
        pbar2.close()
        print('Finished processing.')
        t11 = time.time()
        print(f'{tot_images} images processed in {t11 - t00:0.3f} seconds')

    def process_frames(self, frames):
        p = Images(self.aux_channels)

        for frame in frames:
            images = self.process_frame(frame)
            p.extend(images)

        return p

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
        record = list(data['data'])
        tas = self.tas[frame]

        record_time = datetime.datetime(data['year'],
                                        data['month'],
                                        data['day'],
                                        data['hour'],
                                        data['minute'],
                                        data['second'],
                                        data['ms'] * 1000)
        # if frame % 1000 == 0:
        #     print('At frame ' + str(frame) + ' of ' + str(len(self.data)))
        return self.extract_images(frame, record, record_time, tas)

    def extract_images(self, frame, frame_decomp, record_time, tas):
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
        frequency = self.resolution * 1e-6 / tas

        ii = 0

        syncwords = []
        while ii < len(frame_decomp):
            ii = self.search_syncword(frame_decomp, ii)
            syncwords.append(ii)
            ii += 1

        ref_timeword = None
        timeword_rollovers = 0
        for i, syncword in enumerate(syncwords[:-1]):
            next_syncword = syncwords[i + 1]
            im = frame_decomp[syncword + 1: next_syncword - 1]
            timeword = frame_decomp[next_syncword - 1] & int('0x00ffffff', 16)

            if ref_timeword is None:
                ref_timeword = timeword
                prev_timeword = timeword

            if timeword - prev_timeword < 0:
                timeword_rollovers += 1

            delta_timeword = timeword - ref_timeword \
                + timeword_rollovers * int('0x00ffffff', 16)

            delta_time = delta_timeword * frequency

            image_time = record_time + datetime.timedelta(seconds=delta_time)

            epoch_time = (image_time - self.start_date).total_seconds()
            ns, sec = math.modf(epoch_time)

            image = numpy.array(convert_uint32_to_bitlist(im))
            image = image.flatten()

            if len(image) > 0:
                images.ns.append(ns * 1e9)
                images.sec.append(sec)
                images.length.append(len(im))
                images.image.append(image)
                images.tas.append(tas)
                images.timeword.append(timeword)
                images.buffer_index.append(frame)

            prev_timeword = timeword

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
                i = frame_decomp.index(self.syncword, i)
                if (frame_decomp[i - 1] != int('0xffffffff', 16)) & \
                    (frame_decomp[i - 2] == int('0xffffffff', 16)):
                    return i
                i += 1
        except (ValueError, AttributeError, IndexError) as e:
            return len(frame_decomp)

    def calc_buffer_datetimes(self):
        """ Calculates datetimes from bufffers read in from file and sets
        to datetimes class attribute.
        """
        self.datetimes = [datetime.datetime(d['year'],
                                            d['month'],
                                            d['day'],
                                            d['hour'],
                                            d['minute'],
                                            d['second'],
                                            d['ms'] * 1000) for d in self.data]


