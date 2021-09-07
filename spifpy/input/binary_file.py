#!/usr/bin/env python
# coding: utf-8

import datetime

import numpy

class BinaryFile(object):
    """ Abstract class representing base BinaryFile. Should be subclassed by
    more specific binary file classes.

    Attributes
    ----------
    name : str
        Instrument name of current instrument.
    diodes : int
        Number of diodes in current instrument.
    aux_channels : list
        List of auxiliary channels to create in Particle object, e.g., tas,
        particle_number, etc.
    file_dtype : numpy dtype object
        Numpy dtype object defining binary file structure for specific instrument
        type.
    filename : str
        Filename of current file being referenced.
    data : numpy structured array
        Numpy structured array containing buffers read in from binary file.
    start_date : datetime object
        datetime object containing start date of data in current file.


    Parameters
    ----------
    filename : str
        Filename of current file.
    """

    def __init__(self, filename, inst_name, resolution):
        self.diodes = 0
        self.file_dtype = None
        self.aux_channels = None
        self.data = None
        self.start_date = None
        self.name = inst_name
        self.resolution = int(resolution)

        self.filename = filename

    def read(self):
        """ Reads binary data from file using file_dtype defined by subclass.
        """
        with open(self.filename) as fid:
            self.data = numpy.fromfile(fid, dtype=self.file_dtype)

        self.get_start_date()

        self.calc_buffer_datetimes()

    def get_start_date(self):
        """ Sets start_date attribute with first date found in buffer headers.
        """
        self.start_date = datetime.datetime(self.data['year'][0],
                                            self.data['month'][0],
                                            self.data['day'][0])
    
    def calc_buffer_datetimes(self):
        """ Calculates datetimes from buffers read in from file and sets
        to datetimes class attribute.

        This method is meant to be overriden, but has been reduced to
        calculating datetimes up to a second level of resolution. This is
        because the units past the level of one second might be different,
        (aka one probe could have it in ns, while another could have it
        in ms), so at least when you do call this function for any given
        probe, one can at least use a minimally informative list of datetimes.
        """
        
        self.datetimes = [datetime.datetime(d['year'],
                                            d['month'],
                                            d['day'],
                                            d['hour'],
                                            d['minute'],
                                            d['second']) for d in self.data]