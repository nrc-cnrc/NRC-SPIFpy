#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser
import datetime
import glob
import os
import shutil
import sys
import pathlib as pl

import netCDF4 as nc

import numpy

def addaux():
    parser = get_parser()
    args = parser.parse_args()

    args_checker = ArgsChecker(args)
    args_checker.check_args()

    aux_checker = AuxFileChecker(args.aux_filename, args.config_filename)
    aux_checker.check_aux_file()

    args_transformer = ArgsTransformer(args)

    transformed_args = args_transformer.transform_args()

    add_aux_to_spif(
        transformed_args['spif_filename'],
        transformed_args['aux_filename'],
        transformed_args['inst_names'],
        transformed_args['config_filename'],
        transformed_args['output_filename']
    )

def get_parser():
    parser = argparse.ArgumentParser(description='Adds auxiliary data'
                                                 ' to SPIF file.')

    parser.add_argument('spif_filename',
                        type=str,
                        help='path to SPIF file to modify. ' +
                             'Accepts wildcard characters to process ' +
                             'directories if argument is enclosed in quotes.')

    parser.add_argument('aux_filename',
                        type=str,
                        help='path to source file for aux data')

    parser.add_argument('-i',
                        dest='inst_names',
                        type=str,
                        help='instrument group to apply aux data to -- optional.' +
                             ' If none provided, aux data will be added to all' +
                             ' instruments in SPIF file',
                        default='all')

    parser.add_argument('-c', '--config-filename',
                        dest='config_filename',
                        type=str,
                        help='path to config file for aux data',
                        default=None)

    parser.add_argument('-o',
                        dest='output_filename',
                        type=str,
                        help='when provided, spif file is copied to directory' +
                             ' and aux data is added to new file.',
                        default=None)

    return parser


class ArgsChecker:
    def __init__(self, args):
        self.args = args

    def check_args(self):
        self.check_filename(self.args.spif_filename)

        self.check_aux_filename(self.args.aux_filename)

        if self.args.inst_names != 'all':
            self.check_inst_name(self.args.spif_filename, self.args.inst_names)
        
        self.check_config_filename(self.args.config_filename)
        
        if self.args.output_filename is not None:
            self.check_output_filename(self.args.output_filename)

    def check_filename(self, filename):

        try:
            assert pl.Path(filename).is_file()
        except AssertionError:
            print(f"ERROR : The file {filename} is not a valid file")
            raise

        try:
            assert pl.Path(filename).suffix == '.nc'
        except AssertionError:
            print(f"ERROR : The file {filename} does not have a .nc filename ending")
            raise

        try:
            assert pl.Path(filename).stat().st_size
        except AssertionError:
            print(f"ERROR : The file {filename} is an empty file")
            raise

    def check_aux_filename(self, aux_filename):

        try:
            assert pl.Path(aux_filename).is_file()
        except AssertionError:
            print(f"ERROR : The auxiliary file {aux_filename} is not a valid file")
            raise

        try:
            assert pl.Path(aux_filename).suffix == '.nc'
        except AssertionError:
            print(f"ERROR : The auxiliary file {aux_filename} does not have a .nc filename ending")
            raise

        try:
            assert pl.Path(aux_filename).stat().st_size
        except AssertionError:
            print(f"ERROR : The auxiliary file {aux_filename} is an empty file")
            raise

    def check_inst_name(self, spif_filename, inst_name):
        
        dset = nc.Dataset(spif_filename, mode = 'r', format = 'NETCDF4')

        groups = [x for x in dset.groups.keys()]
        dset.close()
        try:
            assert inst_name in groups
        except AssertionError:
            print(f"ERROR : The provided instrument name {inst_name} is not in the groups" + \
                    f" of the file {spif_filename} : {' | '.join(groups)}")
            raise

    def check_config_filename(self, config_filename):

        try:
            open(config_filename, 'r')
        except OSError:
            print(f"ERROR : The provided output filename {config_filename} is not a valid filename.")
            raise

        try:
            assert pl.Path(config_filename).suffix == '.ini'
        except AssertionError:
            print(f"ERROR : The provided filename {config_filename} does not have a .ini file ending")
            raise

    def check_output_filename(self, output_filename):

        try:
            open(output_filename, 'w')
        except OSError:
            print(f"ERROR : The output file argument {output_filename} is not a valid file")
            raise

        try:
            assert pl.Path(output_filename).suffix == '.nc'
        except AssertionError:
            print(f"ERROR : The output file argument {output_filename} does not have a .nc filename ending")
            raise

class AuxFileChecker:

    def __init__(self, aux_filename, config_filename) -> None:
        self.aux_filename = aux_filename
        self.config_filename = config_filename
        self.config = self.get_config()
        self.aux_var_names = self.get_aux_var_names()

    def get_config(self):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(self.config_filename)

        return config

    def get_aux_var_names(self):
        dset = nc.Dataset(self.aux_filename, mode = 'r', format = 'NETCDF4')
        aux_varnames = [x for x in dset.variables.keys()]
        dset.close()
        return aux_varnames

    def check_aux_file(self):
        self.check_aux_file_open()
        self.check_tas()
        self.check_data_disc()

    def check_aux_file_open(self):
        try:
            dset = nc.Dataset(self.aux_filename, mode = 'r', format = 'NETCDF4')
            dset.close()
        except Exception:
            print(f"The auxiliary file {self.aux_filename} could not be opened for reading.")
            raise

    def check_tas(self):
        orig_tas_name = self.config['aux_data'].get('orig_tas', None)
        corr_tas_name = self.config['aux_data'].get('corrected_tas', None)
        
        try:
            assert orig_tas_name in self.aux_var_names
        except AssertionError:
            print(f"The supplied original TAS name {orig_tas_name} could not be found in the auxiliary file.")
            raise

        try:
            assert corr_tas_name in self.aux_var_names
        except AssertionError:
            print(f"The supplied corrected TAS name {corr_tas_name} could not be found in the auxiliary file.")
            raise

    def check_data_disc(self): 

        for k,v in self.config['data_disc'].items():

            try:
                assert v in self.aux_var_names
            except AssertionError:
                print(f"ERROR : The variable {v} for auxiliary parameter {k} is not present in" + \
                      f" the auxiliary file {self.aux_filename}.")
                raise


class ArgsTransformer:

     def __init__(self, args) -> None:
         self.args = args

         self.transformed_args = {
              'spif_filename':None,
              'aux_filename':None,
              'config':None,
              'inst_names':None,
              'output_filename':None
         }

     def transform_args(self):
          self.transform_filename()
          self.transform_aux_filename()
          self.transform_config()
          self.transform_inst_names()
          self.transform_output_filename()
          
          return self.transformed_args

     def transform_filename(self):
          self.transformed_args['spif_filename'] = pl.Path(self.args.spif_filename)

     def transform_aux_filename(self):
          self.transformed_args['aux_filename'] = pl.Path(self.args.aux_filename)

     def transform_config(self):
          self.transformed_args['config_filename'] = self.args.config_filename

     def transform_inst_names(self):
        dset = nc.Dataset(self.args.spif_filename, mode = 'r', format = 'NETCDF4')

        if self.args.inst_names == 'all':
            inst_names = [k for k in dset.groups.keys()]
        else:
            inst_names = [self.args.inst_names]

        dset.close()

        self.transformed_args['inst_names'] = inst_names

     def transform_output_filename(self):
        if self.args.output_filename == None:
            self.transformed_args['output_filename'] = pl.Path(self.args.spif_filename)
        else:
            self.transformed_args['output_filename'] = pl.Path(self.args.output_filename)


def add_aux_to_spif(spif_filename, aux_file, inst_names, config_file, output_filename):
    """ Function which adds auxiliary data to SPIF NetCDF file from L2
    atmospheric state NetCDF file.

    Parameters
    ----------
    spif_filename : str or Dataset
        Filename or NetCDF dataset of SPIF NetCDF file to modify.
    aux_file : str
        Filename of L2 atmospheric state NetCDF file to take aux data from.
    inst_name : str
        Instrument name to modify in SPIF file.
    config_file : str, optional
        Name of config file containing parameter definitions to copy to SPIF
        file. If provided, other optional arguments are ignored.
        If not provided, orig_tas must be defined.
    output_filename : str, optional
        If provided, SPIF file is copied to specified directory and aux data
        written in new file.
    """

    spif_rootgrp = nc.Dataset(spif_filename, 'r+')

    # Read parameters from config_file

    config = configparser.ConfigParser()
    config.read(config_file)

    aux_config = config['aux_data']
    orig_tas = aux_config['orig_tas']
    corrected_tas = aux_config.get('corrected_tas', None)

    data_disc = config['data_disc']

    for instrument in inst_names:

        print(f'\nAdding aux to {instrument}...')
        
        spif_instgrp = spif_rootgrp[instrument]
        
        if 'aux' not in spif_instgrp.groups:
            spif_instgrp.createGroup('aux')
        spif_auxgrp = spif_instgrp['aux']

        # Get start_date from SPIF File to use in recalculating aux timestamps.
        spif_startdate = datetime.datetime.strptime(
            spif_rootgrp.start_date.strip(), '%Y-%m-%d %H:%M:%S')

        # Open aux file.
        aux_rootgrp = nc.Dataset(aux_file)
        aux_time = aux_rootgrp['Time']

        aux_rootgrp.set_auto_mask(False)

        # Get datetime objects for timestamps in aux_file.
        aux_datetime = convert_netcdf_time(aux_time[:],
                                           aux_time.units)

        # Create AuxTime dimension in SPIF file if it does not exist.
        if 'AuxTime' not in spif_auxgrp.dimensions:
            spif_auxgrp.createDimension(
                'AuxTime', aux_rootgrp.dimensions['Time'].size)

        # Write aux timestamps to SPIF file in aux group.
        write_aux_time(spif_auxgrp, aux_datetime, spif_startdate)

        # Copy original TAS variable to SPIF file.
        copy_aux_variable(spif_auxgrp,
                          'TAS_original',
                          aux_rootgrp,
                          orig_tas)

        # If corrected_tas provided, copy to SPIF file.
        if corrected_tas is not None:
            copy_aux_variable(spif_auxgrp,
                              'TAS_corrected',
                              aux_rootgrp,
                              corrected_tas)

        # If aux_params are provided, copy to SPIF file.
        for dst_name, src_name in data_disc.items():
            copy_aux_variable(spif_auxgrp,
                              dst_name,
                              aux_rootgrp,
                              src_name)

        for key, value in data_disc.items():
                spif_auxgrp.setncattr(key, value)

def write_aux_time(spif_auxgrp, aux_datetime, spif_startdate):
    """ Writes aux timestamps to SPIF file after converting to use same
    reference time.

    Parameters
    ----------
    spif_auxgrp : netCDF group
        Group object pointing to aux group in SPIF NetCDF
    aux_datetime : array of datetime
        Datetimes corresponding to data in aux file.
    spif_startdate : str
        String start_date of SPIF data.
    """
    spif_aux_seconds = nc.date2num(aux_datetime, f'seconds since {spif_startdate}')

    if 'aux_seconds' not in spif_auxgrp.variables:
        fv = numpy.NaN

        zlib = True
        complevel = 4
        out = spif_auxgrp.createVariable('aux_seconds',
                                         'f8',
                                         ('AuxTime',),
                                         fill_value=fv,
                                         zlib=zlib,
                                         complevel=complevel)
    else:
        out = spif_auxgrp['aux_seconds']

    attrs = {'standard_name': 'time',
             'units': 'seconds since start_date',
             'strptime_format': 'seconds since %F %T %z',
             'long_name': 'auxiliary data timestamp'}

    out.setncatts(attrs)
    out[:] = spif_aux_seconds


def copy_aux_variable(spif_auxgrp, spif_aux_var_name, aux_grp, aux_var_name,
    dims=('AuxTime',)):
    """ Given names for source and destination NetCDF groups and variable
    names, copies data from source to destination (creating variables as
    needed).

    Parameters
    ----------
    spif_auxgrp : NetCDF group
        Group object pointing to aux group in SPIF NetCDF
    spif_aux_var_name : str
        Variable name to use for aux data in SPIF file
    aux_grp : NetCDF group
        Group object pointing to source group in aux NetCDF
    aux_var_name : str
        Name of source variable
    dims : tuple, optional
        Dimensions to use for destination variable. Defaults to ('AuxTime',).
    """

    var = aux_grp[aux_var_name]

    if spif_aux_var_name not in spif_auxgrp.variables:
        fv = None
        if '_FillValue' in var.ncattrs():
            fv = var.getncattr('_FillValue')

        zlib = True
        complevel = 4

        out = spif_auxgrp.createVariable(spif_aux_var_name,
                                         var.dtype,
                                         dims,
                                         fill_value=fv,
                                         zlib=zlib,
                                         complevel=complevel,
                                         )
    else:
        out = spif_auxgrp[spif_aux_var_name]

    out_dict = var.__dict__
    out_dict.pop('_FillValue')

    if spif_aux_var_name != aux_var_name:
        out_dict['source_variable'] = aux_var_name

    out.setncatts(out_dict)
    out.setncattr('AuxSource', aux_grp.filepath())

    var_data = var[:]
    out[:] = var_data


def convert_netcdf_time(time, time_basis, start_date=None):
    """ Converts time in standard NetCDF time format (seconds since given
    date-time) into datetime object.

    Parameters
    ----------
    time : array
        Timestamps of data contained within the file. Seconds since
        specified date-time
    time_basis : str
        Format of timestamps, e.g., 'seconds since 2020-07-14 00:00:00'
    start_date : str, optional
        String start date to ues for time_basis, in the event time basis
        is defined in reference to this date.

    Returns
    -------
    list of datetime object
        Absolute datetimes of provided times.
    """

    if start_date is not None:
        time_basis = time_basis.replace('start_date', start_date)

    time_dt = nc.num2date(time, time_basis)

    return time_dt
