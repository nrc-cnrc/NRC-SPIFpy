import configparser
from datetime import datetime
from distutils.command.config import config
import pathlib as pl

from copy import deepcopy

import click

import numpy as np

import netCDF4 as nc
import xarray as xr

from nrc_spifpy.spif import SPIFFile


@click.command()
@click.argument(
    "filename",
    type=click.Path(
        exists=True,  # File needs to exist
        file_okay=True,  # Can provide a file as argument
        dir_okay=False,  # Cannot provide a directory as an argument
        path_type=pl.Path,  # Pre-emptively convert to a pl.Path object
    ),
)
@click.argument(
    "config_file",
    type=click.Path(
        exists=True,  # File needs to exist
        file_okay=True,  # Can provide a file as argument
        dir_okay=False,  # Cannot provide a directory as an argument
        path_type=pl.Path,  # Pre-emptively convert to a pl.Path object
    ),
)
@click.argument("start", type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S"]))
@click.argument("end", type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S"]))
def cut(filename, config_file, start, end):

    # Args checker 

    # Args transformer

    # Processing
    file_cutter = SPIFCutDatasetGenerator(filename, config_file, start, end)

    file_cutter.slice_dataset()

    file_cutter.save()

class SPIFCutDatasetGenerator:
    def __init__(self, filename, config_file, start, end) -> None:

        self.filename = filename
        self.config_file = config_file
        self.start = start
        self.end = end

        self.cut_file = self.create_cut_file()
        self.add_variables_to_cut_file_groups()

    @property
    def cut_filename(self):
        start_str = self.start.strftime("%H%M%S")
        end_str = self.end.strftime("%H%M%S")

        return f"{self.filename.resolve().parent}/{self.config_file.stem}_{start_str}_{end_str}.nc"

    @property
    def file_start_date(self):
        data = nc.Dataset(self.filename, mode="r")
        start_date = getattr(data,"start_date").strip()
        data.close()

        return datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

    @property
    def file_groups(self):
        data = nc.Dataset(self.filename, mode="r")
        groups = [k for k in data.groups.keys()]
        data.close()

        return groups

    @property
    def file_config(self):

        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform = str

        config.read(self.config_file)

        return config

    def create_cut_file(self):
        cut_file = SPIFFile( self.cut_filename, self.file_config )
        cut_file.create_file()

        for g in self.file_groups: 
            cut_file.create_inst_group(g)

        cut_file.set_start_date(self.file_start_date.__str__())

        return cut_file

    def add_variables_to_cut_file_groups(self):
        for g in self.file_groups:
            original_data = xr.open_dataset( self.filename, group = f"{g}/core", decode_cf = False )

            for v in original_data:
                dataset = self.cut_file.rootgrp.groups[g].groups['core']

                if v not in dataset.variables :
                    self.cut_file.create_variable(dataset, v, original_data[v].dtype, original_data[v].dims, attrs=original_data[v].attrs)

            original_data.close()

    def slice_dataset(self):
        for g in self.file_groups:
            cut_group = self.slice_group(g)

            core_grp = self.cut_file.rootgrp.groups[g].groups['core']

            for v in cut_group:
                self.cut_file.write_variable(core_grp, v, cut_group[v].data)

    def slice_group(self, group):
        data = xr.open_dataset(self.filename, group = f"{group}/core", decode_cf = False)

        # datetime to seconds since start date
        imgsec_start = int((self.start-self.file_start_date).total_seconds())
        imgsec_end = int((self.end-self.file_start_date).total_seconds())

        # image start and end indices
        image_start = np.searchsorted(data['image_sec'].values, imgsec_start, side='left')-1
        image_end = np.searchsorted(data['image_sec'].values, imgsec_end, side='right')

        # pixel start and end indices
        pixel_start = int(np.sum(data['image_len'].values[0:image_start])*128)
        pixel_end = int(np.sum(data['image_len'].values[0:image_end-1])*128)

        # buffer start and end indices
        buffer_start = data['buffer_index'].values[image_start]
        buffer_end = data['buffer_index'].values[image_end]

        # cut data
        data_cut = \
            data.isel( Pixels  = slice( pixel_start,  pixel_end  ) ) \
                .isel( Images  = slice( image_start,  image_end  ) ) \
                .isel( Buffers = slice( buffer_start, buffer_end ) )
        
        data_cut.attrs = deepcopy(data.attrs)

        data.close()

        return data_cut

    def save(self):
        self.cut_file.close()

