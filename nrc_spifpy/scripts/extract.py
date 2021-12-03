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

     # Transform the arguments to the proper forms for smooth processing

     args_transformer = ArgsTransformer(args)
     transformed_args = args_transformer.transform_args()

     spif_core = call_spifcore(transformed_args)
     spif_core.process(processors=None)

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

    return parser

def call_spifcore(transformed_args):
     inst_name = get_inst_name(transformed_args)
     filename = transformed_args['filename']
     outfile = transformed_args['output']
     config = transformed_args['config']

     spif_core = SPIFCore(
          inst_dict[inst_name],
          filename,
          outfile,
          config
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

class ArgsTransformer:

     def __init__(self, args) -> None:
         self.args = args

         self.transformed_args = {
              'filename':None,
              'config':None,
              'output':None,
              'aux_file':None,
              'aux_config':None,
         }

     def transform_args(self):
          self.transform_filename()
          self.transform_config()
          self.transform_output()
          
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