#!/usr/bin/env python
# coding: utf-8

import shutil
import argparse
import pathlib as pl

import pkg_resources

def copyconf():
    parser = argparse.ArgumentParser(description='Copies over a configuration file to current working directory')

    parser.add_argument('instrument',
                        type=str,
                        help= 'Name of the instrument to be processed')

    args = parser.parse_args()

    instrument = args.instrument

    instrument_config_name = f"{instrument}.ini"

    try:
        assert pkg_resources.resource_exists('nrc_spifpy', f"config/{instrument_config_name}")
    except AssertionError:
        print(f"ERROR : The supplied instrument name {instrument} cannot be found in the config list.")
    
    instrument_config_data = pkg_resources.resource_string('nrc_spifpy', f"config/{instrument_config_name}")
    aux_config_data = pkg_resources.resource_string('nrc_spifpy', f"config/aux_config.ini")

    dst_root = pl.Path.cwd()

    dst_instrument_config = dst_root / instrument_config_name
    dst_aux_config = dst_root / "aux_config.ini"

    dst_instrument_config.write_bytes(instrument_config_data)
    dst_aux_config.write_bytes(aux_config_data)