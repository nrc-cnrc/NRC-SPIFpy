#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import csv
import datetime
import gc
import math
import os
import time

import numpy
from tqdm import tqdm

from . import BinaryFile
from .spec_utils import process_spec_aux
from ..images import Images

MAX_PROCESSORS = 20


class SPECFile(BinaryFile):
    """ Class representing monoscale binary format for SPEC instruments.
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
        Name of current instrument.
    """
    syncword = numpy.array([3] * 128)

    def __init__(self, filename, inst_name, resolution):
        super().__init__(filename, inst_name, resolution)
        self.diodes = 128
        self.file_dtype = numpy.dtype([('year', 'u2'),
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
        self.aux_channels = ['tas', 'clock_counts']

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
        self.datetimes = numpy.array(self.datetimes)

    def process_file(self, spiffile, processors=None, start=0, end=None):
        """ Method to process file using multiple processors. Calls
        process_image method implemented in children classes. Passes image
        data to spiffile object to be written to file each time 10,000 or more
        images are extracted from file.

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

        if end is None:
            process_until = len(self.data)
        else:
            process_until = end - 1

        data_chunk = range(start, process_until)
        tot_h_images = 0
        tot_v_images = 0

        if self.name == '2DS':
            spiffile.create_inst_group(self.name + '-V')
            spiffile.create_inst_group(self.name + '-H')
            spiffile.set_filenames_attr(self.name + '-V', self.filename)
            spiffile.set_filenames_attr(self.name + '-H', self.filename)
            inst_groups = [self.name + '-H', self.name + '-V']
        elif self.name == 'HVPS4':
            spiffile.create_inst_group(self.name + '-V50', 50)
            spiffile.create_inst_group(self.name + '-V150', 150)
            spiffile.create_inst_group(self.name + '-H50', 50)
            spiffile.create_inst_group(self.name + '-H150', 150)
            spiffile.set_filenames_attr(self.name + '-V50', self.filename)
            spiffile.set_filenames_attr(self.name + '-V150', self.filename)
            spiffile.set_filenames_attr(self.name + '-H50', self.filename)
            spiffile.set_filenames_attr(self.name + '-H150', self.filename)
            inst_groups = [self.name + '-H50', self.name + '-V50',
                           self.name + '-H150', self.name + '-V150']
        else:
            spiffile.create_inst_group(self.name)
            spiffile.set_filenames_attr(self.name, self.filename)
            inst_groups = [self.name]

        spiffile.write_buffer_info(self.start_date, self.datetimes)

        for inst in inst_groups:
            print(f'Extracting HK data for {inst}...')
            process_spec_aux(spiffile,
                             inst,
                             self.data,
                             self.datetimes,
                             self.start_date)
        print('\n')

        if self.name == 'HVPS4':
            hk_file = str(self.filename) + 'HK'
        else:
            hk_file = None

        self.hk_data = FrameInfo(self.data, self.datetimes, hk_file)

        pbar1 = tqdm(desc='Processing frames',
                     total=process_until - start,
                     unit='frame')
        pbar2 = tqdm(desc='Writing frames',
                     total=process_until - start,
                     unit='frame')

        t00 = time.time()
        i = 0
        if self.name == 'HVPS4':
            chunksize = 300
        else:
            chunksize = 500
        max_write_queue = 8
        images_remaining = True
        futures = []

        # self.last_clocks = numpy.zeros(4)
        with ProcessPoolExecutor(max_workers=processors) as executor:
            while True:
                while len(futures) <= max_write_queue and images_remaining:
                    futures.append(executor.submit(self.process_frames,
                                                   data_chunk[i: i + chunksize]))
                    i += chunksize
                    if i >= process_until - start:
                        images_remaining = False
                    pbar1.update(chunksize)

                done, running = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    indx = futures.index(f)
                    if indx == 0:
                        pbar2.update(chunksize)
                        h_images, v_images, h150_images, v150_images, frames = f.result()
                        tot_h_images += len(h_images)
                        tot_v_images += len(v_images)
                        t0 = time.time()
                        self.partial_write(spiffile, h_images, v_images, h150_images, v150_images)

                        # print(f'Frame {frame} images '
                        #       f'written in {time.time() - t0:0.3f} seconds')
                        futures.pop(indx)
                        gc.collect()

                if not images_remaining and len(futures) == 0:
                    break

        pbar1.close()
        pbar2.close()
        print('Finished processing.')
        t0 = time.time()
        t11 = time.time()
        print(f'{tot_h_images}-H; {tot_v_images}-V images processed in {t11 - t00:0.3f} seconds')

        print('Recalculating image times...')

        self.calc_image_times(inst_groups, spiffile)

    def calc_image_times(self, inst_groups, spiffile):
        """
        Recalucates image times based on procedure implemented by Aaron Bansemer
        in SODA2 (see file specnewtime.pro).

        Parameters
        ----------
        inst_groups : list
            List of string names of instrument groups in data file
        spiffile : SPIFFile object
            SPIFFile object of current SPIF NetCDF output file

        """

        # Define needed parameters for recomputing times
        times = numpy.array(self.datetimes, dtype='datetime64[ns]') - numpy.datetime64(self.start_date)
        secs = times.astype('timedelta64[s]')
        ns = times - secs
        datetimes = secs.astype(float) + ns.astype(float) * 1e-9

        # Iterate over instrument groups in current file
        for i, inst_group in enumerate(inst_groups):

            # Read relevant parameters from spiffile
            buffer_indx = spiffile.instgrps[inst_group]['core']['buffer_index'][:]
            try:
                tas = spiffile.instgrps[inst_group]['core']['tas'][:]
            except IndexError:
                continue
            counts = spiffile.instgrps[inst_group]['core']['clock_counts'][:]

            # Get timestamp for each image corresponding their parent buffer
            buffer_time = datetimes[buffer_indx]

            # Calculate delta between counter and buffer
            delta_count = numpy.diff(counts, prepend=counts[0])
            delta_buffer = numpy.diff(buffer_time, prepend=buffer_time[0])

            # Find places where counter rolled over -- should happen every
            # 5.5 mins at 150 m/s
            # Checks both for negative counter value, and large gap in buffer time
            rollover = numpy.where((delta_count < 0) | (delta_buffer > 200))[0]

            # Add rollover max time to get positive delta times
            # HVPS4 has 48 bit counter, all other probes have 32 bit counter
            if self.name == 'HVPS4':
                delta_count[delta_count < 0] += 2 ** 48
            else:
                delta_count[delta_count < 0] += 2 ** 32

            # Convert delta counter to delta time based on airspeed
            delta_time = delta_count * self.resolution / tas * 1e-6

            # Calculate times at the end of each buffer -- this is used below
            # to match buffer times to the partcile time at the end of each
            # buffer
            itimematch = numpy.where(delta_buffer > 0)[0] - 1

            # Use buffer time as best first guess, should be no drift or rollovers
            newtime = buffer_time
            elapsed_time = numpy.zeros(len(newtime))

            # Set rollover indexes for use in loop below. If there were no
            # rollovers, only value needed is the end of the array. Otherwise
            # Each rollover location plus the end of the array is needed.
            if len(rollover) > 0:
                rollover = numpy.append(rollover, [len(newtime)])
            else:
                rollover = [len(newtime)]

            # Loop over all rollover locations to calculate best time within
            # each period.
            istart = 0
            for ro in rollover:
                istop = ro - 1
                # Calculate total elapsed counter time in each rollover period
                elapsed_time[istart: istop] = numpy.cumsum(delta_time[istart: istop])

                # Find buffer boundaries that fall within rollover period
                matches = numpy.where((itimematch > istart) & (itimematch < istop))[0]

                # Find median difference between buffer boundaries and
                # image counter elapsed time within rollover period
                # and add this to elapsed time to calculate new image time
                offset = numpy.median(buffer_time[itimematch[matches]] - elapsed_time[itimematch[matches]])
                newtime[istart: istop] = elapsed_time[istart: istop] + offset
                istart = ro

            # Recalculate seconds and ns from new time
            epoch_time = numpy.modf(newtime)
            secs = epoch_time[1]
            ns = epoch_time[0] * 1e9

            # Save new time to file
            grp = spiffile.instgrps[inst_group]['core']
            spiffile.write_variable(grp, 'image_sec', secs)
            spiffile.write_variable(grp, 'image_ns', ns)

    def partial_write(self, spiffile, h_p, v_p, h_p150=None, v_p150=None):
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
        h_p150 : Images object, optional
            Images object containing processed H array image info with 150µm resolution (in case of HVPS4)
        v_p150 : Images object, optional
            Images object containing processed V array image info with 150µm resolution (in case of HVPS4)
        """

        if self.name == '2DS':
            self._partial_write(spiffile, h_p, '-H')
            v_suffix = '-V'
        elif self.name == 'HVPS4':
            self._partial_write(spiffile, h_p, '-H50')
            self._partial_write(spiffile, h_p150, '-H150')
            self._partial_write(spiffile, v_p150, '-V150')
            v_suffix = '-V50'
        else:
            v_suffix = ''

        self._partial_write(spiffile, v_p, v_suffix)

        v_p.clear()
        h_p.clear()

    def _partial_write(self, spiffile, images, suffix=''):
        if len(images) > 0:
            images.conv_to_array(self.diodes)
            spiffile.write_images(self.name + suffix, images)

    def process_frames(self, frames):
        h_p = Images(self.aux_channels)
        v_p = Images(self.aux_channels)
        h150_p = Images(self.aux_channels)
        v150_p = Images(self.aux_channels)

        prev_counts = {'h': numpy.nan,
                       'v': numpy.nan,
                       'h150': numpy.nan,
                       'v150': numpy.nan}

        h_img = None
        v_img = None
        h_len = 0
        v_len = 0

        h150_img = None
        v150_img = None
        h150_len = 0
        v150_len = 0

        img_dict = {'v_len': v_len,
                    'v150_len': v150_len,
                    'v_img': v_img,
                    'v150_img': v150_img,
                    'h_len': h_len,
                    'h150_len': h150_len,
                    'h_img': h_img,
                    'h150_img': h150_img,
                    'h_rem': 0,
                    'v_rem': 0,
                    'last_h': None,
                    'last_v': None
                    }

        for frame in frames:
            img_dict, prev_counts = self.process_frame(frame, img_dict, prev_counts)
            h_p.extend(img_dict['h_images'])
            v_p.extend(img_dict['v_images'])
            if img_dict['h150_images'] is not None:
                h150_p.extend(img_dict['h150_images'])
                v150_p.extend(img_dict['v150_images'])

        return h_p, v_p, h150_p, v150_p, frames

    def process_frame(self, frame, img_dict, prev_counts):
        """ Method to process single frame of image data. Decompresses image
        data from frame and passes resulting image data to process_image
        method to extract image info from image buffer.

        Parameters
        ----------
        frame : int
            Frame number in self.data to process.
        img_dict : dict
            Dictionary containing image and buffer information that
            spans frames
        prev_counts : dict
            Dictionary containing last image counter information for
            each channel

        Returns
        -------
        dict
            Dictionary containing image and buffer information that
            spans frames
        dict
            Dictionary containing last image counter information for
            each channel
        """
        data = self.data[frame]
        record = data['data']

        # Define Images objects for current frame
        h_images = Images(self.aux_channels)
        v_images = Images(self.aux_channels)
        h150_images = Images(self.aux_channels)
        v150_images = Images(self.aux_channels)

        record_time = datetime.datetime(data['year'],
                                        data['month'],
                                        data['day'],
                                        data['hour'],
                                        data['minute'],
                                        data['second'],
                                        data['ms'] * 1000)

        # Set parameter defaults for current frame
        i = 0

        h_img = img_dict['h_img']
        v_img = img_dict['v_img']
        h_len = img_dict['h_len']
        v_len = img_dict['v_len']

        h150_img = img_dict['h150_img']
        v150_img = img_dict['v150_img']
        h150_len = img_dict['h150_len']
        v150_len = img_dict['v150_len']

        h_rem = 0
        v_rem = 0
        h = None
        v = None

        if self.name == 'HVPS4':
            hk_length = 83
            is_hvps4 = True
        else:
            hk_length = 53
            is_hvps4 = False
        tas = self.hk_data.tas[frame]

        # resolution = self.resolution * 1e-6  # 10µm in meters
        resolution = self.resolution

        while i < len(record) - 5:

            # First check if any H or V images remained unprocessed from
            # last frame due to image spanning frame boundary. If image
            # is present, read remaining bytes of image directly from beginning
            # of current frame and store image.
            if img_dict['h_rem'] > 0:
                if img_dict['last_h'] is not None:
                    h = img_dict['last_h']
                    h['n'] = img_dict['h_rem']
                    h_decomp, h_count, h_rem = self.process_image(record, i, h, is_hvps4)
                    h['rem'] = 0
                    if h_count == 0:
                        h_count = h['count']
                    if self.name == 'HVPS4' and h['fifo_array'] == 0:
                        h150_img, h150_len, h150_images = self.store_image(h,
                                                                           h150_img,
                                                                           h_decomp,
                                                                           h150_len,
                                                                           h150_images,
                                                                           record_time,
                                                                           frame,
                                                                           tas,
                                                                           h_count)

                    else:
                        h_img, h_len, h_images = self.store_image(h,
                                                                  h_img,
                                                                  h_decomp,
                                                                  h_len,
                                                                  h_images,
                                                                  record_time,
                                                                  frame,
                                                                  tas,
                                                                  h_count)
                i += img_dict['h_rem']
                img_dict['h_rem'] = 0
            elif img_dict['v_rem'] > 0:
                if img_dict['last_v'] is not None:
                    v = img_dict['last_v']
                    v['n'] = img_dict['v_rem']
                    v_decomp, v_count, v_rem = self.process_image(record, i, v, is_hvps4)
                    # print(v_count)
                    if v_count == 0:
                        v_count = v['count']
                    v['rem'] = 0
                    if self.name == 'HVPS4' and v['fifo_array'] == 0:
                        v150_img, v150_len, v150_images = self.store_image(v,
                                                                           v150_img,
                                                                           v_decomp,
                                                                           v150_len,
                                                                           v150_images,
                                                                           record_time,
                                                                           frame,
                                                                           tas,
                                                                           v_count)

                    else:
                        v_img, v_len, v_images = self.store_image(v,
                                                                  v_img,
                                                                  v_decomp,
                                                                  v_len,
                                                                  v_images,
                                                                  record_time,
                                                                  frame,
                                                                  tas,
                                                                  v_count)
                i += img_dict['v_rem']
                img_dict['v_rem'] = 0
            elif record[i] == 12883:  # equals '2S'
                h = self.decode_flags(record[i + 1])
                v = self.decode_flags(record[i + 2])
                image_count = record[i + 3]
                num_slices = record[i + 4]

                i += 5

                if (h['mismatch'] == 1) or (v['mismatch'] == 1):
                    # print('mismatch')
                    i += h['n'] + v['n']
                else:
                    if h['n'] > 0: # Check if images are present in H buffer
                        h_decomp, h_count, h_rem = self.process_image(record, i, h, is_hvps4)

                        # Store dummy time for now since we will recompute
                        # following batch processing
                        image_time = record_time

                        # If probe is HVPS4 and fifo_array flag is 0, current
                        # image is part of the coarser array
                        if self.name == 'HVPS4' and h['fifo_array'] == 0:
                            if h_count == 0:
                                h_count = prev_counts['h150']

                            h150_img, h150_len, h150_images = self.store_image(h,
                                                                               h150_img,
                                                                               h_decomp,
                                                                               h150_len,
                                                                               h150_images,
                                                                               image_time,
                                                                               frame,
                                                                               tas,
                                                                               h_count)
                            prev_counts['h150'] = h_count
                            h['count'] = h_count

                        else:
                            if h_count == 0:
                                h_count = prev_counts['h']
                            h_img, h_len, h_images = self.store_image(h,
                                                                      h_img,
                                                                      h_decomp,
                                                                      h_len,
                                                                      h_images,
                                                                      image_time,
                                                                      frame,
                                                                      tas,
                                                                      h_count)

                            prev_counts['h'] = h_count
                            h['count'] = h_count

                    if v['n'] > 0: # Check if image are present in V buffer
                        v_decomp, v_count, v_rem = self.process_image(record, i, v, is_hvps4)

                        # Store dummy time for now since we will recompute
                        # following batch processing
                        image_time = record_time

                        # If probe is HVPS4 and fifo_array flag is 0, current
                        # image is part of the coarser array
                        if self.name == 'HVPS4' and v['fifo_array'] == 0:
                            if v_count == 0:
                                v_count = prev_counts['v150']
                            v150_img, v150_len, v150_images = self.store_image(v,
                                                                               v150_img,
                                                                               v_decomp,
                                                                               v150_len,
                                                                               v150_images,
                                                                               image_time,
                                                                               frame,
                                                                               tas,
                                                                               v_count)
                            prev_counts['v150'] = v_count
                            v['count'] = v_count

                        else:
                            if v_count == 0:
                                v_count = prev_counts['v']
                            v_img, v_len, v_images = self.store_image(v,
                                                                      v_img,
                                                                      v_decomp,
                                                                      v_len,
                                                                      v_images,
                                                                      image_time,
                                                                      frame,
                                                                      tas,
                                                                      v_count)
                            prev_counts['v'] = v_count
                            v['count'] = v_count
                            # if frame == 119072 or frame == 119073:
                            #     print(len(v_decomp) / 128, num_slices, v['n'], i + v['n'], v, v_len, len(record) - (i + v['n']))

                    i += h['n'] + v['n']
            elif record[i] == 19787:  # equals 'MK'
                i += 23
            elif record[i] == 18507:  # equals 'HK'
                if i + 50 < len(record):
                    read_tas = numpy.array([record[i + 50], record[i + 49]],
                                           dtype='u2').view('float32')[0]
                    tas_dec = read_tas % 1
                    if read_tas < 1000 and read_tas > 0.1 and tas_dec == 0:
                        tas = read_tas
                    i += hk_length
                else:
                    i += 1
            elif record[i] == 20044:  # equals 'NL'
                break
            else:
                i += 1

        # Store image parameters for use in subsequent data frames.
        # *_images is the set of complete images to store in the parent function
        # *_img is the current 'working' incomplete image
        # *_len is the length in slices of the current working mage
        # *_rem is the number of bytes of working image cut off by end of frame
        # last_* is the last state of the image flag dictionary for the given channel
        img_dict = {'v_images': v_images,
                    'v150_images': v150_images,
                    'v_len': v_len,
                    'v150_len': v150_len,
                    'v_img': v_img,
                    'v150_img': v150_img,
                    'h_images': h_images,
                    'h150_images': h150_images,
                    'h_len': h_len,
                    'h150_len': h150_len,
                    'h_img': h_img,
                    'h150_img': h150_img,
                    'h_rem': h_rem,
                    'v_rem': v_rem,
                    'last_h': h,
                    'last_v': v,
                    }

        return img_dict, prev_counts

    def decode_flags(self, record):
        """ Decode flags for given 16 bit record.

        Parameters
        ----------
        record : int
            16 bit record to extract flags from.

        Returns
        -------
        dict
            Dictionary object containing flags results for given record. Flags
            are:
                n - number of words, including timing words
                timing - 0 if timing words present, 1 if not found
                mismatch - timing word mismatch
                fifo_array - FIFO empty before timing word found (means image
                             was cut off); contains array flag for HVPS4
                overload - last two words of the record are overload timing words
        """
        flags = {}
        flags['n'] = record & 4095
        flags['timing'] = (record & (2 ** 12)) >> 12
        flags['mismatch'] = (record & (2 ** 13)) >> 13
        flags['fifo_array'] = (record & (2 ** 14)) >> 14
        flags['overload'] = (record & (2 ** 15)) >> 15
        flags['slice_rem'] = []
        flags['rem'] = 0

        return flags

    def process_image(self, record, i, p, hvps4=False):
        """ Extracts image from record.

        Parameters
        ----------
        record : array
            Array of records in current buffer.
        i : int
            Current position in record array for processing.
        p : dict
            flags dict for current image
        hvps4 : bool, optional
            Flag to indicate if probe is HVPS4. Default is False.

        Returns
        -------
        array
            Decompressed image.
        int
            Value of counter, if present. If not present, 0 is returned.
        int
            Number of bytes of image that are cutoff by end of current frame
        """
        min_length = 2
        if hvps4:
            min_length += 1

        p_raw = record[i: i + p['n']]

        if i + p['n'] > len(record):
            p_rem = i + p['n'] - len(record)
        else:
            p_rem = 0

        if p['timing'] == 0 and len(p_raw) > min_length and len(record) >= i + p['n']:
            if hvps4:
                p_counter = (p_raw[-1] << 32) | (p_raw[-2] << 16) | p_raw[-3]
                p_raw = p_raw[:-3]
            else:
                p_counter = (p_raw[-2] << 16) | p_raw[-1]
                p_raw = p_raw[:-2]
        else:
            p_counter = 0

        if p['overload'] == 1:
            p_overload = (p_raw[-2] << 16) | p_raw[-1]
            # print(p_overload, p_counter)
            p_raw = p_raw[:-2]

        p_decomp, slice_rem = self.decompress_image(p_raw, hvps4, p_rem, p['slice_rem'])

        p['slice_rem'] = slice_rem
        p['rem'] = p_rem

        return p_decomp, p_counter, p_rem

    def store_image(self, p, p_img, p_decomp, p_len, images, p_time, frame, tas, clock_counts):
        """ If timeword present in current image, stores extracted image
        info in Images data object. Otherwise, concatenates current image
        data to existing image data.

        Parameters
        ----------
        p : dict
            flags dict for current image
        p_img : array
            Image array to write to file if timing word is present.
        p_decomp : array
            Current image array to append to p_img.
        p_len : int
            Length in slices of p_img.
        images : Images object
            Images object to save image data to.
        p_time : datetime object
            Time of current image.
        frame : int
            Buffer frame number currently being processed.
        tas : float
            True airspeed of current image.
        clock_counts : int
            Clock counter for current image.

        Returns
        -------
        array
            Image array of combined images (if no image written to Images
            object), or None if image was written.
        int
            Current length of image array.
        Images object
            Images object updated with new image information if timeword
            was present.
        """
        if p_img is None:
            p_img = p_decomp
            p_len = len(p_img) / 128
        else:
            p_img = numpy.concatenate((p_img, p_decomp), axis=0)
            p_len = len(p_img) / 128

        if p['timing'] == 0 and p['rem'] == 0:
            if p_len > 0:
                epoch_time = math.modf((p_time - self.start_date).total_seconds())
                images.image.append(p_img)
                images.sec.append(epoch_time[1])
                images.ns.append(epoch_time[0] * 1e9)
                images.length.append(p_len)
                images.buffer_index.append(frame)
                images.tas.append(tas)
                images.clock_counts.append(clock_counts)
            p_img = None
            p_len = 0

        return p_img, p_len, images

    def decompress_image(self, img, hvps4, p_rem, slice_decomp):
        """ Decompresses image image.

        Parameters
        ----------
        img : array
            Compressed image data.
        hvps4 : bool
            True if probe is HVPS4
        p_rem : int
            Number of bytes of image that are cutoff by end of current frame
        slice_decomp : array
            Leftover portion of decompressed slice information to be completed in
            later decompression calls.

        Returns
        -------
        array
            Array of decompressed image.
        array
            Leftover portion of decompressed slice information to be completed in
            later decompression calls.
        """
        img_decomp = []
        startslice = 0
        timeslice = 0
        non_compressed = 0
        for line in img:
            if non_compressed > 0:
                bin_line = [-1 * (int(n) - 1) for n in bin(line)[2:].zfill(16)[::-1]]
                slice_decomp.extend(bin_line)
                non_compressed -= 1
                if non_compressed == 0:
                    img_decomp, slice_decomp = self.add_img_slice(img_decomp, slice_decomp)
            elif line == 32767:  # special case of 0x7fff
                img_decomp, slice_decomp = self.add_img_slice(img_decomp, slice_decomp)
                if hvps4:
                    non_compressed += 8
                else:
                    img_decomp.extend([0] * 128)
            elif line == 16384:  # special case of 0x4000
                img_decomp, slice_decomp = self.add_img_slice(img_decomp, slice_decomp)
                img_decomp.extend([1] * 128)
            else:
                timeslice = (line & (2 ** 15)) >> 15
                startslice = (line & (2 ** 14)) >> 14
                num_shaded = (line & 16256) >> 7
                num_clear = (line & 127)
                if timeslice == 0:
                    if startslice == 1:
                        img_decomp, slice_decomp = self.add_img_slice(img_decomp, slice_decomp)
                    slice_decomp.extend([0] * num_clear)
                    slice_decomp.extend([1] * num_shaded)

        if len(slice_decomp) > 0 and p_rem == 0 and timeslice == 0:
            img_decomp, slice_decomp = self.add_img_slice(img_decomp, slice_decomp)
        img = numpy.logical_not(numpy.array(img_decomp))
        return img, slice_decomp

    def add_img_slice(self, img_decomp, slice_decomp):
        if len(slice_decomp) % 128 > 0:
            slice_decomp.extend([0] * (128 - (len(slice_decomp) % 128)))
        img_decomp.extend(slice_decomp)
        slice_decomp = []

        return img_decomp, slice_decomp


class FrameInfo(object):

    def __init__(self, data, datetimes, hk_filename=None):

        n_frames = len(data)

        self.data = data
        self.datetimes = datetimes

        self.tas = numpy.ones(n_frames, dtype=float) * 0.1
        self.last_h_timestamp = numpy.ones(n_frames) * numpy.nan
        self.last_v_timestamp = numpy.ones(n_frames) * numpy.nan

        self.raw_counts = numpy.empty(n_frames, dtype=object)
        self.raw_counts[...] = [[] for i in range(n_frames)]

        if hk_filename is None:
            self.process_hk()
        else:
            self.process_hk_file(hk_filename)

    def process_hk(self):
        last_good_tas = numpy.nan
        for i, frame in enumerate(self.data):
            hk_indx = numpy.where(frame['data'] == 18507)[0]
            record = frame['data']
            tas_vals = []
            for indx in hk_indx:
                try:
                    tas = numpy.array([record[indx + 50],
                                       record[indx + 49]],
                                      dtype='u2').view('float32')[0]
                    tas_dec = tas % 1
                    if tas > 0.1 and tas < 1000 and tas_dec == 0:
                        tas_vals.append(tas)

                except IndexError:
                    pass

            if tas_vals:
                self.tas[i] = tas_vals[-1]
                last_good_tas = tas_vals[-1]
            else:
                self.tas[i] = last_good_tas

    def process_hk_file(self, filename):

        with open(filename) as fid:
            data = numpy.fromfile(fid, dtype='u2')

            possible_hk_index = numpy.where(data == 18507)[0]
            # print(len(possible_hk_index))

            tas = []
            hk_datetimes = []

            for indx in possible_hk_index:
                if data[indx - 8] > 2050 or data[indx - 8] < 1980:
                    continue
                record_time = datetime.datetime(data[indx - 8],
                                                data[indx - 7],
                                                data[indx - 5],
                                                data[indx - 4],
                                                data[indx - 3],
                                                data[indx - 2],
                                                data[indx - 1] * 1000)
                hk_datetimes.append(record_time)

                tas.append(numpy.array([data[indx + 76],
                                        data[indx + 75]],
                                       dtype='u2').view('float32')[0])

        datetimes = numpy.array(self.datetimes).astype('datetime64[us]')
        hk_datetimes = numpy.array(hk_datetimes).astype('datetime64[us]')
        for i, record_time in enumerate(datetimes):
            best_indx = numpy.where(hk_datetimes > record_time)[0]
            # print(best_indx[0], len(self.tas), len(tas), i, len(datetimes))
            if len(best_indx) > 0:
                self.tas[i] = tas[best_indx[0]]
            elif i > 0:
                self.tas[i] = self.tas[i - 1]
            else:
                self.tas[i] = 0


class SpecHKFile(object):

    def __init__(self, filename):
        self.filename = filename
        self.process()

    def process(self):
        with open(self.filename) as fid:
            data = numpy.fromfile(fid, dtype='u2')

        possible_hk_index = numpy.where(data == 18507)[0]
        print(len(possible_hk_index))

        self.hk_timestamps = []
        self.tas = []

        for indx in possible_hk_index:
            record_time = datetime.datetime(data[indx - 8],
                                            data[indx - 7],
                                            data[indx - 5],
                                            data[indx - 4],
                                            data[indx - 3],
                                            data[indx - 2],
                                            data[indx - 1] * 1000)
            self.hk_timestamps.append(record_time)
            self.tas.append(numpy.array([data[indx + 76],
                                         data[indx + 75]],
                                        dtype='u2').view('float32')[0])

        self.timestamps = numpy.array(self.hk_timestamps)
        self.tas = numpy.array(self.tas)


class SpecHKData(object):

    def __init__(self, data):
        self.data = data
        self.process()

    def process(self):
        frame_tas_vals = []
        last_good_tas = numpy.nan
        for frame in self.data:
            hk_indx = numpy.where(frame['data'] == 18507)[0]
            record = frame['data']
            tas_vals = []
            for i, indx in enumerate(hk_indx):
                try:
                    tas = numpy.array([record[indx + 50],
                                       record[indx + 49]],
                                      dtype='u2').view('float32')[0]
                    tas_dec = tas % 1
                    if tas > 0.1 and tas < 1000 and tas_dec == 0:
                        tas_vals.append(tas)

                except IndexError:
                    pass

            if tas_vals:
                frame_tas_vals.append(tas_vals[-1])
                last_good_tas = tas_vals[-1]
            else:
                frame_tas_vals.append(last_good_tas)

        self.tas = numpy.array(frame_tas_vals)