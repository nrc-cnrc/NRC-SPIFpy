#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import datetime
import gc
import math
import os
import time

import numpy
from tqdm import tqdm

from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.images import Images

MAX_PROCESSORS = 20

from pprint import pprint

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
        self.aux_channels = [
            'tas',
            'num_words',
            'timing_flag',
            'mismatch_flag',
            'fifo_flag',
            'overload_flag',
            'particle_count',
            'num_slices',
            'timing_word_upper',
            'timing_word_lower'
        ]

        self.aux_dtypes = {
            'tas':'f',
            'num_words':'u2',
            'timing_flag':'u2',
            'mismatch_flag':'u2',
            'fifo_flag':'u2',
            'overload_flag':'u2',
            'particle_count':'u2',
            'num_slices':'u2',
            'timing_word_upper':'u2',
            'timing_word_lower':'u2'
        }

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

    def process_file(self, spiffile, processors=None):
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

        process_until = len(self.data)

        data_chunk = range(0, process_until)
        tot_h_images = 0
        tot_v_images = 0

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
                     total=process_until,
                     unit='frame')
        pbar2 = tqdm(desc='Writing frames',
                     total=process_until,
                     unit='frame')

        i = 0
        chunksize = 500
        images_remaining = True

        while images_remaining:
            h_images, v_images, frame = self.process_frames(data_chunk[i:i + chunksize])

            i += chunksize
            
            if i >= process_until:
                images_remaining = False
            
            pbar1.update(chunksize)

            self.partial_write(spiffile, h_images, v_images)

            pbar2.update(chunksize)

            tot_h_images += len(h_images)
            tot_v_images += len(v_images)

        pbar1.close()
        pbar2.close()

    def partial_write(self, spiffile, h_p, v_p):
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
                spiffile.write_images_with_extra_aux_dtypes(self.name + '-H', h_p, self.aux_dtypes)
            v_suffix = '-V'
        else:
            v_suffix = ''

        if len(v_p) > 0:
            v_p.conv_to_array(self.diodes)
            spiffile.write_images_with_extra_aux_dtypes(self.name + v_suffix, v_p, self.aux_dtypes)

        v_p.clear()
        h_p.clear()

    def process_frames(self, frames):
        h_p = Images(self.aux_channels)
        v_p = Images(self.aux_channels)

        for frame in frames:
            h_images, v_images = self.process_frame(frame)
            h_p.extend(h_images)
            v_p.extend(v_images)

        return h_p, v_p, frames[0]

    def process_frame(self, frame):
        """ Method to process single frame of image data. Decompresses image
        data from frame and passes resulting image data to process_image
        method to extract image info from image buffer.

        Parameters
        ----------
        frame : int
            Frame number in self.data to process.

        Returns
        -------
        Images object
            Images object containing H array image info in current frame.
        Images object
            Images object containing V array image info in current frame.
        """
        data = self.data[frame]
        record = data['data']

        # Define Images objects for current frame
        h_images = Images(self.aux_channels)
        v_images = Images(self.aux_channels)


        # Set parameter defaults for current frame
        i = 0
        h_img = None
        v_img = None
        h_len = 0
        v_len = 0

        tas = 100  # m/s
        resolution = 10 * 1e-6  # 10µm in meters
        tick_length = resolution / tas

        record_time = datetime.datetime(data['year'],
                                        data['month'],
                                        data['day'],
                                        data['hour'],
                                        data['minute'],
                                        data['second'],
                                        data['ms'] * 1000)
        while i < len(record) - 53:
            if record[i] == 12883:  # equals '2S'
                h = self.decode_flags(record[i + 1])
                v = self.decode_flags(record[i + 2])
                image_count = record[i + 3]
                num_slices = record[i + 4]
                
                i += 5
                if (h['mismatch'] == 1) or (v['mismatch'] == 1):
                    i += h['n'] + v['n']
                else:
                    if h['n'] > 0:
                        h_decomp, h_timing_word_upper, h_timing_word_lower = self.process_image(record, i, h)
                        #time_delta = h_count * tick_length
                        #image_time = record_time + datetime.timedelta(seconds=int(time_delta))
                        h_img, h_len, h_images = self.store_image(p = h,
                                                                  p_img = h_img,
                                                                  p_decomp = h_decomp,
                                                                  p_len = h_len,
                                                                  images = h_images,
                                                                  record_time = record_time,
                                                                  timing_word_upper = h_timing_word_upper,
                                                                  timing_word_lower = h_timing_word_lower,
                                                                  particle_count = image_count,
                                                                  num_slices = num_slices,
                                                                  frame = frame,
                                                                  tas = tas
                                                                  )

                    if v['n'] > 0:
                        v_decomp, v_timing_word_upper, v_timing_word_lower = self.process_image(record, i, v)
                        #time_delta = v_count * tick_length
                        #try:
                        #    image_time = record_time + datetime.timedelta(seconds=int(time_delta))
                        #except OverflowError as e:
                        #    raise e
                        v_img, v_len, v_images = self.store_image(p = v,
                                                                  p_img = v_img,
                                                                  p_decomp = v_decomp,
                                                                  p_len = v_len,
                                                                  images = v_images,
                                                                  record_time = record_time,
                                                                  timing_word_upper = v_timing_word_upper,
                                                                  timing_word_lower = v_timing_word_lower,
                                                                  particle_count = image_count,
                                                                  num_slices = num_slices,
                                                                  frame = frame,
                                                                  tas = tas
                                                                  )
                    i += h['n'] + v['n']
            elif record[i] == 19787:  # equals 'MK'
                i += 23
            elif record[i] == 18507:  # equals 'HK'
                tas = numpy.array([record[i + 50], record[i + 49]],
                                  dtype='u2').view('float32')[0]
                i += 53
            else:
                i += 1

            if tas > 0.1:
                tick_length = resolution / tas
            else:
                tick_length = 0

        return h_images, v_images

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
                fifo - FIFO empty before timing word found (means image
                       was cut off)
                overload - last two words of the record are overload timing words
        """
        flags = {}
        flags['n'] = record & 4095
        flags['timing'] = (record & (2 ** 12)) >> 12
        flags['mismatch'] = (record & (2 ** 13)) >> 13
        flags['fifo'] = (record & (2 ** 14)) >> 14
        flags['overload'] = (record & (2 ** 15)) >> 15

        return flags

    def process_image(self, record, i, p):
        """ Extracts image from record.

        Parameters
        ----------
        record : array
            Array of records in current buffer.
        i : int
            Current position in record array for processing.
        p : dict
            flags dict for current image

        Returns
        -------
        array
            Decompressed image.
        int
            Value of counter, if present. If not present, 0 is returned.
        """
        p_raw = record[i: i + p['n']]
        if p['timing'] == 0 and len(p_raw) > 2:
            timing_upper = p_raw[-2]
            timing_lower = p_raw[-1]
            p_raw = p_raw[:-2]
        else:
            timing_upper = 0
            timing_lower = 0
        p_decomp = self.decompress_image(p_raw)

        return p_decomp, timing_upper, timing_lower

    def store_image(self, p, p_img, p_decomp, p_len, images, record_time, timing_word_upper, timing_word_lower, particle_count, num_slices, frame, tas):
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

        if p['timing'] == 0:
            if p_len > 0:
                epoch_time = math.modf((record_time - self.start_date).total_seconds())
                images.image.append(p_img)
                images.sec.append(epoch_time[1])
                images.ns.append(epoch_time[0] * 1e9)
                images.length.append(p_len)
                images.buffer_index.append(frame)
                images.tas.append(tas)
                images.num_words.append(p['n'])
                images.timing_flag.append(p['timing'])
                images.mismatch_flag.append(p['mismatch'])
                images.fifo_flag.append(p['fifo'])
                images.overload_flag.append(p['overload'])
                images.particle_count.append(particle_count)
                images.num_slices.append(num_slices)
                images.timing_word_upper.append(timing_word_upper)
                images.timing_word_lower.append(timing_word_lower)

            p_img = None
            p_len = 0

        return p_img, p_len, images

    def decompress_image(self, img):
        """ Decompresses image image.

        Parameters
        ----------
        img : array
            Compressed image data.

        Returns
        -------
        array
            Array of decompressed image image.
        """
        img_decomp = []
        slice_decomp = []
        for line in img:
            if line == 32767:  # special case of 0x7fff
                img_decomp.extend([0] * 128)
            elif line == 16384:  # special case of 0x4000
                img_decomp.extend([1] * 128)
            else:
                timeslice = (line & (2 ** 15)) >> 15
                startslice = (line & (2 ** 14)) >> 14
                num_shaded = (line & 16256) >> 7
                num_clear = (line & 127)
                #print(line, timeslice, startslice, num_clear, num_shaded)
                if timeslice == 0:
                    if startslice == 1:
                        if len(slice_decomp) % 128 > 0:
                            slice_decomp.extend([0] * (128 - (len(slice_decomp) % 128)))
                        img_decomp.extend(slice_decomp)
                        slice_decomp = []

                    # if num_clear + num_shaded < 128:
                    slice_decomp.extend([0] * num_clear)
                    slice_decomp.extend([1] * num_shaded)

        img = numpy.logical_not(numpy.array(img_decomp))
        return img


