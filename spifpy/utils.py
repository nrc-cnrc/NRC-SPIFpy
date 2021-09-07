#!/usr/bin/env python
# coding: utf-8

import math

from numba import njit
import numpy


def convert_datetimes_to_seconds(start_date, datetimes):
    """ Converts difference in datetimes to total elapsed seconds.

    Parameters
    ----------
    start_date : datetime object
        Start date to use for calculating total elapsed seconds.
    datetimes : list of datetimes
        List of datetimes to calculate total elapsed seconds for.

    Returns
    -------
    list of float
        Total elapsed seconds from start_date for each element in datetimes list.
    """
    datetimes_seconds = [(dt - start_date).total_seconds() for dt in datetimes]

    return datetimes_seconds


def bin_to_int(bin_list, inverse=True):
    """ Given arbitrary length list of binary values, calculate integer value.

    Parameters
    ----------
    bin_list : list of int
        List of binary values for calculating integer value.
    inverse : bool, optional
        If true (default) caluclate integer from bin_list assuming first element
        is LSB. If false, integer will be calculated from bin_list assuming first
        element is MSB.

    Returns
    -------
    int
        Integer value calculated from bin_list.
    """
    if inverse:
        step = -1
    else:
        step = 1
    bin_list = bin_list[::step]
    bin_int = 0
    for bit in bin_list:
        bin_int = (bin_int << 1) | bit
    return int(bin_int)


def list_to_array(lst, diodes):
    """ Convert list of ragged length 1-D arrays to square numpy array.

    Parameters
    ----------
    lst : list of 1-D arrays
        List of arrays to normalize
    diodes : int
        Number of diodes in instrument array.

    Returns
    -------
    array
        Square 2-d array of input arrays.
    int
        Maximum length of any individual 1-D array, scaled to next factor
        of n-diodes to allow reshaping at later step.
    """
    lengths = numpy.array([len(item) for item in lst])
    max_length = math.ceil(lengths.max() / diodes) * diodes
    mask = lengths[:, None] > numpy.arange(max_length)
    out = numpy.zeros(mask.shape, dtype='uint8')
    out[mask] = numpy.concatenate(lst)
    return out, max_length


def list3d_to_array(lst, diodes):
    """ Convert list of ragged length 2-D arrays to 3-D numpy array.

    Parameters
    ----------
    lst : list of 2-D arrays
        List of arrays to normalize
    diodes : int
        Number of diodes in instrument array.

    Returns
    -------
    array
        3-d array of input arrays.
    """
    lengths = numpy.array([len(item) for item in lst])
    max_length = lengths.max()
    mask = lengths[:, None] > numpy.arange(max_length)
    out = numpy.zeros((mask.shape[0], mask.shape[1], diodes), dtype='uint8')
    out[mask] = numpy.concatenate(lst)
    return out


def convert_uint8_to_bitlist(image):
    """ Converts uint8 value to list of bits assuming little endianness.

    Parameters
    ----------
    image : int
        Int value to convert to binary.

    Returns
    -------
    array
        Array of bits corresponding to provided uint8 value.
    """
    img = numpy.array(image, dtype=numpy.uint8)
    bin_img = numpy.unpackbits(img, bitorder='little')
    return bin_img


def convert_uint32_to_bitlist(image):
    if type(image) is list:
        bin_img = []
        for im in image:
            bin_img.append(_convert_uint32_to_bitlist(im))
    else:
        bin_img = _convert_uint32_to_bitlist(image)
    return bin_img


def _convert_uint32_to_bitlist(image):
    bitstr = f'{image:032b}'
    bin_img = [int(b) for b in list(bitstr)]
    return bin_img


@njit
def _reshape_image(image, max_slices, diodes):
    """ Numba reshape image function for faster processing.
    """
    return numpy.reshape(image, (-1, max_slices // diodes, diodes))
