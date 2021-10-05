#!/usr/bin/env python
# coding: utf-8

import datetime

import numpy

from  import BinaryFile

MAX_PROCESSORS = 20

class SPECFile(BinaryFile):
    """ Class representing monoscale binary format for SPEC instruments.
    Implements methods specific to decompressing and processing images from
    this type of binary file.

    Parameters
    ----------
    filename : str
        Filename of current file.
    inst_name : str
        Name of current instrument.
    """

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
        self.aux_channels = ['tas']

    def calc_buffer_datetimes(self):
        """ Calculates datetimes from buffers read in from file and sets
        to datetimes class attribute.
        """
        self.datetimes = [datetime.datetime(d['year'],
                                            d['month'],
                                            d['day'],
                                            d['hour'],
                                            d['minute'],
                                            d['second'],
                                            d['ms'] * 1000) for d in self.data]

def process_file(self, spiffile):
    spiffile.set_start_date(self.start_date.strftime('%Y-%m-%d %H:%M:%S %z'))


# NOTE : All words in a particle frame are 16 bits

"""
    1. When the FIFO is nearly full the PD2FIFO CPLD will go into overload and will write the timing word of the overload start time into their FIFOs.  When the overload period ends the overload timing word is written to the FIFO.  The overflow timing words are written in a particle record that has no image data (slices = 0 and NH or NV = 2).
    
    2. When in STEREO mode the probe will record a single particle event that extends across as many particle frames as required until matching timing words are found in both channels.
    
    3. When in stereo mode the channels will enter and leave overload at the same time so only one set of timing words is sent to the data system.
"""

class N_HV_Word_Processor:
    """
    This is for words 2 in a particle frame

    NH (Word 2)
    -----------------------------------------------------------
    Bits 0 –11 Number of horizontal words–Includes Timing Words if present
    Bit 12 –1 = Timing Words not found 
    Bit 13 –Timing Word mismatch
    Bit 14—FIFO Empty (means the next particle was cut off)
    Bit 15 –The last two words of the horizontal data record are overload timing words

    NV (Word 3)
    -------------------------------------------------------------
    Bits 0 –11 Number of vertical words–Includes Timing Words if not same as  the  horizontal Timing Word and the TW were found.
    Bit 12 –1 = Timing Words not found 
    Bit 13 –Timing Word mismatch 
    Bit 14-FIFO Empty before timing word found
    Bit 15 –The last two words of the vertical data record are overload timing words
    """

    def __init__(self, word) -> None:
        self.word = word
    
    @property
    def num_h_or_v_words(self):
        # Bit masking out of a 16-bit number
        # to only get the 12 bit component
        return self.word & 0b0000111111111111

    @property
    def timing_words_not_found(self):
        # Bitmask to get 12th bit only, then bit shift right
        # 12 spots to keep only that bit
        return (self.word & 0b0001000000000000) >> 12

    @property
    def timing_word_mismatch(self):
        # Bitmask to get 13th bit only, then bit shift right
        # 13 spots to keep only that bit
        return (self.word & 0b0010000000000000) >> 13

    @property
    def fifo_empty(self):
        # Bitmask to get 14th bit only, then bit shift right
        # 14 spots to keep only that bit
        return (self.word & 0b0100000000000000) >> 14

    @property
    def overload_timing_words_exist(self):
        # Bitmask to get 15th bit only, then bit shift right
        # 15 spots to keep only that bit
        return (self.word & 0b1000000000000000) >> 15

class Particle_Count_Processor:  
    """
        This is for word 4 in a particle frame

        Particle Count –Total number of particles events detected.
            1. If timing words are not found, the next particle record will be a continuation of this record and thus the particle count will remain the same for all particle records that make up the particle event

            2. In stereo mode, a particle event will continue until timing words are found in both channels.

            3. In BOTH mode a separate particle count and slices processed through the current record is maintained for each channel
    """

    def __init__(self, word) -> None:
        self.word = word

    @property
    def particle_count(self):
        return self.word

class Num_Slices_Processor:
    """ 
        This is for word 5 in a particle frame

        Number of slices in this particle.  During a multi-particle record event this parameter will be the number of slices detected during the event through the current record.  In STEREO mode, the number of slices reported is from the horizontal channel.  By the end of the event the number of vertical slices will always equal the number of horizontal slices.  In BOTH mode this will represent the number of slices in a particle event for the channel that contains data.
    """

    def __init__(self, word) -> None:
        self.word = word

    @property
    def num_slices(self):
        return self.word

class Image_Data_Parser:

    """
    
    This is for the actual image data itself

    -----------------------------------------------------------|
    Bit Number |Value Description                              |
    -----------------------------------------------------------|
    15         |  0    Indicates that this is an image word    |
    -----------------------------------------------------------|           
    14         |  1    Indicates the start of a slice (FIRST)  |
               |  0    Indicates the continuation of a slice   |
    -----------------------------------------------------------|
    7 -13      |       Number of shaded pixels (bit 13 is MSB) |
    0 -6       |       Number of clear pixels (bit 6 is MSB)   |
    -----------------------------------------------------------|

    0x7fff = 128 clear bits
    0x4000 = 128 set bits
    
    """

    def __init__(self, data) -> None:
        self.data = data

class Timing_Word_Parser:
    """
    This is for timing words at the end of image data

    Unless bit 12 of the horizontal length is set, the last two words of the horizontal image data are the timing words that are generated by a free running counter in the PD2FIFO CPLD.  This counter increments once every slice interval regardless of the state of the compression state machine (overload or normal compression mode).
    
    Word 1 -Bits 16 -31 of the counter 
    Word 2 -Bits 0 -15 of the counter

    Unless bit 12 of the vertical length is set, the last two words of the vertical image data are the timing words that are generated by a free running counter in the PD2FIFO CPLD.  This counter increments once every slice interval regardless of the state of the compression state machine (overload or normal compression mode).  When the probe is operating in the stereo mode this timing word is NOT present as the vertical timer MUST equal the horizontal timer.
    
    Word 1 -Bits 16 -31 of the counter 
    Word 2 -Bits 0 -15 of the counter
    """

    def __init__(self, word1, word2) -> None:
        self.word1 = word1
        self.word2 = word2