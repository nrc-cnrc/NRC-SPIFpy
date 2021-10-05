#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import datetime
import os
import time

from tqdm import tqdm

from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.input.input_utils import read_sea_image_data
from nrc_spifpy.images import Images

MAX_PROCESSORS = 20


class DMTBinaryFile(BinaryFile):
    """ Abstract class representing generic DMT binary file. Contains method
    used to process images using multiple processors.
    """

    def __init__(self, filename, inst_name, resolution):
        super().__init__(filename, inst_name, resolution)

    def read(self):
        file_ext = os.path.splitext(self.filename)[1]

        if file_ext == '.sea':
            self.read_sea()
        else:
            super().read()

    def read_sea(self, typ='mono'):

        self.data = read_sea_image_data(
            self.filename, self.file_dtype, typ, self.name)

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
                     total=len(self.data),
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
                        images.conv_to_array(self.diodes)
                        spiffile.write_images(self.name, images)

                        futures.pop(indx)

                if not images_remaining and len(futures) == 0:
                    break

        pbar1.close()
        pbar2.close()
        t11 = time.time()
        print(f'{tot_images} images processed in {t11 - t00:0.3f} seconds')

    def process_frames(self, frames):
        p = Images(self.aux_channels)

        for frame in frames:
            images = self.process_frame(frame)
            p.extend(images)

        return p

    def process_frame(self, frame):
        raise NotImplementedError

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

