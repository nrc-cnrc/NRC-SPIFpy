#!/usr/bin/env python
# coding: utf-8

import datetime
import math

import numpy as np

from tqdm import tqdm

from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.input.dmt.mono.buffer import decompress_dmt_mono_buffer
from nrc_spifpy.input.dmt.mono.image import find_syncword_sequences, get_image_boundaries
from nrc_spifpy.input.dmt.mono.image import ImageMetadataProcessor, ImageDataProcessor, ImageAssembler, AssembledImageContainer
from nrc_spifpy.input.dmt.mono.image import add_auxiliary_core_variables

file_buffer_template = np.dtype([
    ('year', 'u2'),
    ('month', 'u2'),
    ('day', 'u2'),
    ('hour', 'u2'),
    ('minute', 'u2'),
    ('second', 'u2'),
    ('ms', 'u2'),
    ('weekday', 'u2'),
    ('data', '(4096, )B')
])

syncword = 0xAA

sync_sequence = np.array([syncword] * 8, dtype='B')

class DMTMonoFile(BinaryFile):
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

    def __init__(self, filename, inst_name, resolution):
        super().__init__(filename, inst_name, resolution)
        self.diodes = 64
        self.file_dtype = file_buffer_template

    def process_file(self, spiffile, processors = None):
        """
        Parameters
        ----------
        spiffile : SPIFFile object
            SPIFFile object of current SPIF NetCDF output file
        """

        spiffile.set_start_date(self.start_date.strftime('%Y-%m-%d %H:%M:%S %z'))

        spiffile.create_inst_group(self.name)
        spiffile.write_buffer_info(self.start_date, self.datetimes)
        spiffile.set_filenames_attr(self.name, self.filename)

        add_auxiliary_core_variables(spiffile, self.name)

        data_processor = DMTMonoDataProcessor(self.data['data'])

        processed_data = data_processor.process_data()

        self.write_image_data(spiffile, self.name, processed_data)

    def write_image_data(self, spiffile, instrument, images):

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
        spiffile.write_variable(coregrp, 'buffer_index', images.buffer_index, dim_size)
        spiffile.write_variable(coregrp, 'image', images.image, px_size)

        # Auxiliary SPIF variables

        spiffile.write_variable(coregrp, 'dof_flag', images.dof_flag, dim_size)
        spiffile.write_variable(coregrp, 'particle_count', images.particle_count, dim_size)
        spiffile.write_variable(coregrp, 'slice_count', images.slice_count, dim_size)

        spiffile.rootgrp.sync()

class DMTMonoDataProcessor:

    def __init__(self, data) -> None:
        self.data = data
        self.decompressed_data = np.array([], dtype='B')
        self.image_bounds = np.array([], dtype = 'i4')
        self.image_metadata_containers = []
        self.image_data = []

    def process_data(self):
        decompressed_data = self.decompress_data()
        image_bounds = self.get_image_bounds(decompressed_data)

        metadata_containers = self.process_image_metadata(decompressed_data, image_bounds)
        image_containers = self.process_image_data(decompressed_data, image_bounds)

        assembled_image_containers = self.assemble_image_data(metadata_containers, image_containers)

        return assembled_image_containers

    def decompress_data(self):
        pbar = tqdm(desc='Decompressing buffers',
             total=len(self.data),
             unit='buffers')


        decompressed_data = np.array([],dtype = 'B')

        for buffer in self.data:
            decompressed_buffer = decompress_dmt_mono_buffer(buffer)
            decompressed_data = np.append(decompressed_data, decompressed_buffer)
            pbar.update(1)

        pbar.close()

        return decompressed_data

    def get_image_bounds(self, decompressed_data):
        syncword_sequence_inds = find_syncword_sequences(decompressed_data)
        image_bounds = get_image_boundaries(decompressed_data, syncword_sequence_inds)

        return image_bounds

    def process_image_metadata(self, decompressed_data, image_bounds):
        pbar = tqdm(
            desc='Extracting image metadata',
            total=len(image_bounds),
            unit='images'
        )

        metadata_containers = [None]*len(image_bounds)

        metadata_processor = ImageMetadataProcessor()

        for i in range(len(image_bounds)):
            image = decompressed_data[image_bounds[i,0]:image_bounds[i,1]]
            image = image[8:] # cut out the string of 0xAA's that indicate the start of an image
            metadata_containers[i] = metadata_processor.process_image_metadata(image)
            pbar.update(1)

        pbar.close()

        return metadata_containers

    def process_image_data(self, decompressed_data, image_bounds):
        pbar = tqdm(
            desc='Extracting images',
            total=len(image_bounds),
            unit='images'
        )

        image_containers = [None]*len(image_bounds)

        image_processor = ImageDataProcessor()

        for i in range(len(image_bounds)):
            image = decompressed_data[image_bounds[i,0]:image_bounds[i,1]]
            image = image[8:] # cut out the string of 0xAA's that indicate the start of an image
            image_containers[i] = image_processor.process_image_data(image)
            pbar.update(1)

        pbar.close()

        return image_containers

    def assemble_image_data(self, metadata_containers, image_containers):

        assembled_image_container = AssembledImageContainer()

        image_assembler = ImageAssembler()

        assembled_image_container = image_assembler.assemble_images(metadata_containers, image_containers)

        return assembled_image_container