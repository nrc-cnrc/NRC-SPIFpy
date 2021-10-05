#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser
import re
import os
import sys
import pathlib as pl

import datetime as dt

from nrc_spifpy.input import DMTMonoFile
from nrc_spifpy.input import DMTGreyFile
from nrc_spifpy.input import SPECFile
from nrc_spifpy.input import TwoDFile
from nrc_spifpy.spif import SPIFCore

inst_dict = {'2DC': TwoDFile,
             '2DP': TwoDFile,
             'CIP': DMTMonoFile,
             'CIPGS': DMTGreyFile,
             'PIP': DMTMonoFile,
             '2DS': SPECFile,
             'HVPS': SPECFile}

def extract():

     # Get the parser and grab the args

     parser = get_parser()
     args = parser.parse_args()

     # Check the args, make sure everything is OK before using
     # the arguments to do anything with them

     args_checker = ArgsChecker(args)
     args_checker.check_args()

     # Do some more detailed checking of the times

     time_checker = SPIFStartEndTimeChecker(args.filename, args.config)
     time_checker.check_time_args(args.start, args.end)

     # Transform the arguments to the proper forms for smooth processing

     args_transformer = ArgsTransformer(args)
     transformed_args = args_transformer.transform_args()

     spif_core = call_spifcore(transformed_args)
     spif_core.process(processors=transformed_args['nproc'])

def get_parser():
    parser = argparse.ArgumentParser(description='Processes raw OAP data to' +
                                                 ' SPIF formatted NetCDF file')
    parser.add_argument('filename',
                        type=str,
                        help= 'path to raw instrument file to process.')

    parser.add_argument('config',
                        type=str,
                        help= 'path to config file to use for processing')

    parser.add_argument('-o',
                        dest='output',
                        type=str,
                        help='Filename to use for SPIF output',
                        default=None)

    parser.add_argument('--start',
                        dest='start',
                        type=str,
                        help='The start time in the file to begin processing.',
                        default=None)

    parser.add_argument('--end',
                        dest='end',
                        type=str,
                        help='The end time in the file to begin processing.',
                        default=None)

    parser.add_argument('-n',
                        dest='nproc',
                        type=int,
                        help='number of processors to use in processing',
                        default=None)

    return parser

def call_spifcore(transformed_args):
     inst_name = get_inst_name(transformed_args)
     filename = transformed_args['filename']
     outfile = transformed_args['output']
     config = transformed_args['config']
     start_time = transformed_args['start']
     end_time = transformed_args['end']

     spif_core = SPIFCore(
          inst_dict[inst_name],
          filename,
          outfile,
          config,
          start_time = start_time,
          end_time = end_time
     )

     return spif_core

def get_inst_name(transformed_args):
     config = configparser.ConfigParser(allow_no_value=True)

     config.read(transformed_args['config'])
     inst_name = config['instrument'].get('instrument_name', None)

     return inst_name

class ArgsChecker:
     def __init__(self, args):
          self.args = args

     def check_args(self):
          self.check_filename(self.args.filename)
          self.check_config(self.args.config)

          if self.args.output is not None : self.check_output(self.args.output)
          if self.args.start is not None : self.check_time_args(self.args.start)
          if self.args.end is not None : self.check_time_args(self.args.end)
          if self.args.nproc is not None : self.check_nproc(self.args.nproc)
          
     def check_filename(self, filename):

          try:
               assert pl.Path(filename).is_file()
          except AssertionError:
               print(f"ERROR : The file {filename} is not a valid file")
               raise

          try:
               assert pl.Path(filename).stat().st_size
          except AssertionError:
               print(f"ERROR : The file {filename} is an empty file")
               raise

     def check_config(self, config_file):
          config = configparser.ConfigParser(allow_no_value=True)

          config.read(config_file)
          inst_name = config['instrument'].get('instrument_name', None)

          try:
               assert inst_name in inst_dict
          except AssertionError:
               print(f"ERROR : The provided instrument type {inst_name} from config file {config} is invalid. Please provide a valid instrument from the following list {' | '.join([k for k in inst_dict.keys()])}")
               raise

     def check_output(self, output):

          try:
               open(output, 'w')
          except OSError:
               print(f"ERROR : The provided output filename {output} is not a valid filename.")
               raise

          try:
               assert pl.Path(output).suffix == '.nc'
          except AssertionError:
               print(f"ERROR : The provided filename {output} does not have a .nc file ending")
               raise

     def check_time_args(self, time):
          try:
               assert re.match('[0-9]{2}:[0-9]{2}:[0-9]{2}', time)
          except AssertionError:
               print(f"ERROR : The provided time {time} does not match the required format of HH:MM:SS")

          split_time = [int(x) for x in time.split(':')]

          try:
               assert split_time[0] < 24 and split_time[0] >= 0
               assert split_time[1] < 60 and split_time[0] >= 0
               assert split_time[2] < 60 and split_time[0] >= 0
          except AssertionError:
               print(f"ERROR : The provided time {time} does not have proper digits")
               raise

     def check_nproc(self, nproc):
          try:
               assert isinstance(nproc, int)
          except AssertionError:
               print(f"ERROR : The provided argument for nproc {nproc} is not a number.")
               raise

          try:
               assert nproc > 0
          except AssertionError:
               print(f"ERROR : The provided argument nproc {nproc} is less than one")
               raise

class SPIFStartEndTimeChecker:

     def __init__(self, filename, config_file):
          self.filename = filename
          self.config_file = config_file

          self.start_time = None
          self.end_time = None
     
          self.get_start_end_time()

     def get_start_end_time(self):
          config = configparser.ConfigParser(allow_no_value=True)

          config.read(self.config_file)
          inst_name = config['instrument'].get('instrument_name', None)

          inst_class = inst_dict[inst_name]

          self.inst_file = inst_class(
               self.filename,
               config['instrument']['instrument_name'],
               config['resolution']['value']
          )

          self.inst_file.read()

          file_datetimes = self.inst_file.datetimes

          self.file_time_start = self.inst_file.start_date - file_datetimes[0].replace(microsecond = 0)
          self.file_time_end = file_datetimes[-1].replace(microsecond = 0) - self.inst_file.start_date

     def check_time_args(self, start_time, end_time):
          if start_time is not None:
               start_time_sec = [int(x) for x in start_time.split(':')]
               start_time_sec = dt.timedelta(
                    seconds = start_time_sec[0]*3600 + start_time_sec[1]*60 + start_time_sec[2]
               )

          if end_time is not None:
               end_time_sec = [int(x) for x in end_time.split(':')]
               end_time_sec = dt.timedelta(
                    seconds = end_time_sec[0]*3600 + end_time_sec[1]*60 + end_time_sec[2]
               )
          
          if (start_time is not None) and (end_time is not None):
               try:
                    assert start_time_sec < end_time_sec
               except AssertionError:
                    print(f"ERROR : Start time {start_time} is greater than end time {end_time}")
                    raise
          
          if start_time is not None:
               try:
                    self.check_time_in_file(start_time_sec)
               except AssertionError:
                    print(f"ERROR : Start time {start_time} is not in the file time range")
                    raise

          if end_time is not None:
               try:
                    self.check_time_in_file(end_time_sec)
               except AssertionError:
                    print(f"ERROR : End time {end_time} is not in the file time range")
                    raise

     def check_time_in_file(self, time):
          assert time >= self.file_time_start
          assert time <= self.file_time_end

class ArgsTransformer:

     def __init__(self, args) -> None:
         self.args = args

         self.transformed_args = {
              'filename':None,
              'config':None,
              'output':None,
              'start':None,
              'end':None,
              'nproc':None,
              'aux_file':None,
              'aux_config':None,
         }

     def transform_args(self):
          self.transform_filename()
          self.transform_config()
          self.transform_output()
          self.transform_start()
          self.transform_end()
          self.transform_nproc()
          
          return self.transformed_args

     def transform_filename(self):
          self.transformed_args['filename'] = pl.Path(self.args.filename)

     def transform_config(self):
          self.transformed_args['config'] = self.args.config

     def transform_output(self):
          if self.args.output == None:
               input = pl.Path(self.args.filename)
               self.transformed_args['output'] = input.parent / (input.name.replace('.','_') + '.nc')
          else:
               self.transformed_args['output'] = pl.Path(self.args.output)

     def transform_start(self):
          if self.args.start is None:
               self.transformed_args['start'] = None
          else:
               start = [int(x) for x in self.args.start.split(':')]
               start = start[0]*3600 + start[1]*60 + start[2]
               self.transformed_args['start'] = dt.timedelta(seconds = start)

     def transform_end(self):
          if self.args.end is None:
               self.transformed_args['end'] = None
          else:
               end = [int(x) for x in self.args.end.split(':')]
               end = end[0]*3600 + end[1]*60 + end[2]
               self.transformed_args['end'] = dt.timedelta(seconds = end)

     def transform_nproc(self):
          self.transformed_args['nproc'] = self.args.nproc