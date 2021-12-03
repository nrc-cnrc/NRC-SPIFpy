#!/usr/bin/env python
# coding: utf-8

import configparser
import datetime
import os

import netCDF4 as nc
import numpy

from .utils import convert_datetimes_to_seconds

spif_version = '0.86'

TIME_CHUNK = 64 * 32
IMAGE_TIME_CHUNK = 64 * 8

class SPIFCore(object):
    """ Defines toplevel SPIF object, intended to pass information between
    binary file classes and SPIFFile class.

    Attributes
    ----------
    config : configparser object
        Attributes read from instrument ini for use in defining SPIF NetCDF
        file metadata.
    instfile : BinaryFile object
        Reference to BinaryFile type object representing binary file to be
        converted to SPIF format.
    spiffile : SPIFFile object
        Reference to SPIFFile object representing output file for converted
        binary data.

    Parameters
    ----------
    inst_class : BinaryFile class
        Specific child class of BinaryFile to use for reading and converting
        binary data.
    filename : str
        Path to binary file to convert to SPIF format.
    outfile : str
        Path of SPIF output file to create.
    config : str
        Path to config file to use for processing binary file.
    """

    def __init__(self, inst_class, filename, outfile, config, start_time = None, end_time = None):
        self.config = self.get_config(config)

        self.instfile = inst_class(filename,
                                   self.config['instrument']['instrument_name'],
                                   self.config['resolution']['value'])
        self.spiffile = SPIFFile(outfile,
                                 self.config)

        self.start_time = start_time
        self.end_time = end_time

    def process(self, processors=None):
        """ Launches class methods to 1) create SPIF output file; 2) read
        binary file blocks; 3) process binary file blocks and output to SPIF;
        and 4) close SPIF file.
        """
        self.spiffile.create_file()
        self.instfile.read()

        self.instfile.process_file(self.spiffile, processors)
        self.spiffile.close()

    def get_config(self, config_file):
        """ Given config filename, open and return configparser object.

        Parameters
        ----------
        config_file : str
            Path to config file. None can be passed if filename is given.

        Returns
        -------
        configparser object
            ConfigParser object containing the config parameters in ini file.

        Raises
        ------
        FileNotFoundError
            Error raised if:
                - given config file does not exist
                - no config file or filename passed in
                - default config file cant be found in path for given filename
        """
        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform = str

        if config_file is not None:
            if not os.path.exists(config_file):
                raise FileNotFoundError(
                    'Config file could not be found at indicated path.')
        else:
            if self.filename is None:
                raise FileNotFoundError('Config file has not been defined.')
            else:
                config_file = os.path.dirname(
                    self.filename) + os.path.sep + self.name + '.ini'
                if not os.path.exists(config_file):
                    raise FileNotFoundError(
                        'Config file could not be found at default file path.')

        config.read(config_file)

        return config


class SPIFFile(object):
    """ Class representing NetCDF file conforming to SPIF conventions.

    Class Attributes
    ----------
    image_ns_attrs : dict
        Dictionary containing standard attributes for image_ns variable.
    image_sec_attrs : dict
        Dictionary containing standard attributes for image_sec variable.
    image_image_attrs : dict
        Dictionary containing standard attributes for image_image variable.
    image_len_attrs : dict
        Dictionary containing standard attributes for image_len variable.
    buffer_sec_attrs : dict
        Dictionary containing standard attributes for buffer_sec variable.
    inst_scalars : dict
        Dictionary containing names and variable types for instrument-level
        scalar attributes.

    Attributes
    ----------
    outfile : str
        Name of output NetCDF file.
    attrs : configparser object
        ConfigParser object containing attributes read from ini file.
    root_attrs : configparser section
        'root' section taken from attrs attribute.
    instrgrps : dict
        Dictionary for each instrument in SPIFFile containing instrument name
        and reference to its group in the NetCDF file.

    Parameters
    ----------
    outfile : str
        Name of output NetCDF file.
    attrs : configparser object
        ConfigParser object containing attributes read from ini file.

    """

    image_ns_attrs = {'long_name': 'image arrival time in nanoseconds',
                      'units': 'ns since image_sec',
                      'ancillary_variables': 'image_sec'}

    image_sec_attrs = {'standard_name': 'time',
                       'long_name': 'image arrival time in seconds',
                       'timezone': 'UTC',
                       'units': 'seconds since start_date',
                       'strftime_format': '%F %T %z'}
    image_attrs = {'long_name': 'image array'}
    image_len_attrs = {'long_name': 'image length',
                       'units': 'number of slices'}

    buffer_sec_attrs = {'long_name': 'buffer time in seconds',
                        'timezone': 'UTC',
                        'units': 'seconds since start_date',
                        'strftime_format': '%F %T %z'}

    inst_scalars = {'pixels': 'i2',
                    'resolution': 'f4',
                    'resolution_err': 'f4',
                    'arm_separation': 'f4',
                    'arm_separation_err': 'f4',
                    'antishatter_tips': 'b',
                    'bpp': 'i2'}

    def __init__(self, outfile, attrs):

        self.outfile = outfile
        self.attrs = attrs
        self.root_attrs = attrs['root']
        self.root_attrs['conventions'] = f'SPIF-{spif_version}'
        self.root_attrs['history'] = datetime.datetime.now().isoformat()
        self.instgrps = {}

    def create_file(self, mode='w'):
        """ Creates SPIF NetCDF output file and sets root attributes.

        Parameters
        ----------
        mode : str, optional
            File creation mode for NetCDF file.
        """
        self.rootgrp = nc.Dataset(self.outfile, mode, format='NETCDF4')

        self.rootgrp.setncatts(self.root_attrs)

    def create_inst_group(self, inst_name):
        """ Given desired instrument name, creates instrument group and
        associated attributes and variables in NetCDF file.

        Parameters
        ----------
        inst_name : str
            Short name of instrument to use for group name in NetCDF file.
        """
        instgrp = self.rootgrp.createGroup(inst_name)
        self.instgrps[inst_name] = instgrp
        coregrp = instgrp.createGroup('core')
        auxgrp = instgrp.createGroup('aux')

        n_pixels = self.attrs.getint('pixels', 'value')

        instgrp.createDimension('Images', None)
        instgrp.createDimension('Buffers', None)
        instgrp.createDimension('Pixels', None)

        instgrp.setncatts(self.attrs['instrument'])
        for var, var_type in self.inst_scalars.items():
            if self.attrs.has_section(var):
                value = self.attrs[var]['value']
                if var == 'antishatter_tips':
                    value = bool(value)
                attrs = dict(self.attrs[var].items())
                attrs.pop('value')
                self.create_variable(instgrp,
                                     var,
                                     var_type,
                                     (),
                                     attrs,
                                     data=value)

        self.create_variable(coregrp,
                             'image_ns',
                             'i4',
                             ('Images',),
                             self.image_ns_attrs,
                             chunksizes=(TIME_CHUNK,))

        self.create_variable(coregrp,
                             'image_sec',
                             'i4',
                             ('Images',),
                             self.image_sec_attrs,
                             chunksizes=(TIME_CHUNK,))

        self.create_variable(coregrp,
                             'image',
                             'u1',
                             ('Pixels',),
                             self.image_attrs,
                             complevel=3,
                             chunksizes=(IMAGE_TIME_CHUNK * 8 * n_pixels,))

        self.create_variable(coregrp,
                             'image_len',
                             'u2',
                             ('Images'),
                             self.image_len_attrs,
                             chunksizes=(TIME_CHUNK,))

        self.create_variable(coregrp,
                             'buffer_index',
                             'u4',
                             ('Images'),
                             chunksizes=(TIME_CHUNK,))

        self.create_variable(coregrp,
                             'buffer_sec',
                             'i4',
                             ('Buffers',),
                             self.buffer_sec_attrs,
                             chunksizes=(TIME_CHUNK,))

    def set_start_date(self, start_date):
        """ Given reference time, sets start_date attribute in NetCDF file.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD HH:MM:SS Z of data being written to file.

        """
        self.rootgrp.setncattr('start_date', start_date)

    def set_filenames_attr(self, inst_name, filenames):
        if type(filenames) is list:
            filenames = ', '.join(os.path.abspath(filenames))
        else:
            filenames = os.path.abspath(filenames)

        self.instgrps[inst_name].setncattr('raw_filenames', filenames)

    def create_variable(self, dataset, name, dtype, dims, attrs=None,
                        data=None, zlib=True, complevel=3, chunksizes=None):
        """ Given a NetCDF dataset object, output name, data type, and
        NetCDF dimensions, create variable in given NetCDF dataset. If
        data object is passed, write the data out to the created variable
        in the NetCDF file.

        Parameters
        ----------
        dataset : NetCDF dataset or group object
            Dataset to write data to - can be root or group dataset in a NetCDF
            file.
        name : str
            Name to use for variable in NetCDF dataset.
        dtype : str
            Data type of data
        dims : tuple
            Tuple containing string names of NetCDF dimensions to use in for
            data.
        attrs : dict, optional
            Dictionary of attribute name/value pairs to assign to variable.
        data : scalar or array, optional
            Data to write to created variable in NetCDF file.
        zlib : bool, optional
            Flag to activate zlib compression on variable. Default is True.
        complevel : int, optional
            Compression level to use if zlib is true. Defaults to 3, max is 9.
        chunksizes : tuple, optional
            Tuple of chunksizes (one for each dimension) to use in creation
            of variable in NetCDF file.
        """

        if name == '':
            return

        fv = None
        if attrs is not None:
            if '_FillValue' in attrs:
                fv = numpy.array(attrs.pop('_FillValue'), dtype)

        i = 1
        basename = name
        while name in dataset.variables:
            name = basename + '_' + str(i)
            i += 1

        out = dataset.createVariable(name,
                                     dtype,
                                     dims,
                                     fill_value=fv,
                                     zlib=zlib,
                                     complevel=complevel,
                                     chunksizes=chunksizes)

        if attrs is not None:
            for key, val in attrs.items():
                if type(val) is bool:
                    attrs[key] = str(val)
            out.setncatts(attrs)

        if data is not None:
            out[:] = data

    def write_variable(self, dataset, name, data, start_val=None):
        """ Writes data to given variable in provided dataset.

        Parameters
        ----------
        dataset : NetCDF dataset or group object
            Dataset to write data to - can be root or group dataset in a NetCDF
            file.
        name : str
            Name of variable to write to.
        data : scalar or array
            Data to write to file
        start_val : int, optional
            Index of value to use for writing data, e.g. to append data to
            existing variable.
        """
        out = dataset[name]
        if start_val is not None:
            out[start_val:] = data
        else:
            out[:] = data

    def write_images(self, inst_name, images):
        """ Writes image information to NetCDF file. If image information
        already exists in NetCDF file, new image information is appended
        to the end of existing data.

        Parameters
        ----------
        inst_name : str
            Instrument name to write image information to.
        images : Images object
            Images object containing image information to write to file.
        """
        instgrp = self.instgrps[inst_name]
        coregrp = instgrp['core']
        dim_size = len(instgrp.dimensions['Images'])
        px_size = len(instgrp.dimensions['Pixels'])
        self.write_variable(coregrp, 'image_sec', images.sec, dim_size)
        self.write_variable(coregrp, 'image_ns', images.ns, dim_size)
        self.write_variable(coregrp, 'image_len', images.length, dim_size)
        self.write_variable(coregrp, 'buffer_index', images.buffer_index, dim_size)
        self.write_variable(coregrp, 'image', images.image, px_size)
        for key, val in images.__dict__.items():
            if key not in images.default_items:
                if key not in coregrp.variables.keys():
                    self.create_variable(coregrp,
                                         key,
                                         'f',
                                         ('Images',),
                                         attrs=None,
                                         data=val,
                                         zlib=True,
                                         chunksizes=(TIME_CHUNK,))
                else:
                    self.write_variable(coregrp, key, val, dim_size)

        self.rootgrp.sync()

    def write_images_with_extra_aux_dtypes(self, inst_name, images, aux_dtypes):
        """ Writes image information to NetCDF file. If image information
        already exists in NetCDF file, new image information is appended
        to the end of existing data.

        Parameters
        ----------
        inst_name : str
            Instrument name to write image information to.
        images : Images object
            Images object containing image information to write to file.
        """
        instgrp = self.instgrps[inst_name]
        coregrp = instgrp['core']
        dim_size = len(instgrp.dimensions['Images'])
        px_size = len(instgrp.dimensions['Pixels'])
        self.write_variable(coregrp, 'image_sec', images.sec, dim_size)
        self.write_variable(coregrp, 'image_ns', images.ns, dim_size)
        self.write_variable(coregrp, 'image_len', images.length, dim_size)
        self.write_variable(coregrp, 'buffer_index', images.buffer_index, dim_size)
        self.write_variable(coregrp, 'image', images.image, px_size)
        for key, val in images.__dict__.items():
            if key not in images.default_items:
                if key not in coregrp.variables.keys():
                    self.create_variable(coregrp,
                                         key,
                                         aux_dtypes[key],
                                         ('Images',),
                                         attrs=None,
                                         data=val,
                                         zlib=True,
                                         chunksizes=(TIME_CHUNK,))
                else:
                    self.write_variable(coregrp, key, val, dim_size)

        self.rootgrp.sync()

    # def write_aux(self, instgrp, aux_dim_name, aux_dim_length, aux_dict):

    #     if 'Aux' in instgrp.groups:
    #         auxgrp = instgrp['Aux']
    #     else:
    #         auxgrp = instgrp.createGroup('Aux')

    #     auxgrp.createDimension(aux_dim_name, (aux_dim_length,))

    #     for key, val in aux_dict.items():
    #         self.create_variable(auxgrp,
    #                              key,
    #                              'f',
    #                              (aux_dim_name,),
    #                              zlib=True
    #                              )

    def write_buffer_info(self, start_date, datetimes):
        """ Writes buffer header information to core group of NetCDF file.

        Parameters
        ----------
        start_date : datetime object
            Reference time to use for converting datetimes into seconds since
            start_date.
        datetimes : array of datetime objects
            Datetimes read from binary file buffer headers.
        """
        seconds = convert_datetimes_to_seconds(start_date, datetimes)
        for grp in self.instgrps.values():
            auxgrp = grp['core']
            self.write_variable(auxgrp, 'buffer_sec', seconds)

    def close(self):
        """ Closes NetCDF file.
        """
        self.rootgrp.close()

