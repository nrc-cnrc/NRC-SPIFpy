#!/usr/bin/env python
# coding: utf-8

import gc
import time

import numpy

from .utils import list_to_array, list3d_to_array, _reshape_image


class Images(object):
    """ Class containing info for set of images. Default info includes
    nanoseconds, seconds, image, length, and buffer index. Users can add other
    parameters by passing list at initialization with names of other parameters
    to store.

    Class Attributes
    ----------------
    default_items : list of str
        List of default items common to all image sources.

    Attributes
    ----------
    ns : list
        Nanoseconds timestamp for images.
    sec : list
        Seconds timestamp for images.
    image : list
        Image data for images.
    length : list
        Length data, in slices, for images.
    buffer_index : list
        Buffer index value for images.

    Parameters
    ----------
    aux_names : list of str, optional
        List of strings of auxiliary parameters to include in current Images
        instance. If provided, strings in list become class attributes.
    """
    default_items = ['ns', 'sec', 'image', 'length', 'buffer_index']

    def __init__(self, aux_names=None):
        for item in self.default_items:
            setattr(self, item, [])
        if aux_names is not None:
            for name in aux_names:
                setattr(self, name, [])

    def __len__(self):
        return len(self.sec)

    def clear(self):
        """ Resets all class attributes to empty lists.
        """
        for key in self.__dict__.keys():
            setattr(self, key, [])
        gc.collect()

    def extend(self, p):
        """ Extends all attributes in current instance with corresponding
        attributes in given instance.

        Parameters
        ----------
        p : Images object
            Images object to copy data from to current instance.
        """
        if p is not None:
            if len(p) > 0:
                for key, val in self.__dict__.items():
                    getattr(self, key).extend(getattr(p, key))

    def reshape_image(self, diodes):
        """ Reshapes image data to 3-D array, with dimensions of image, slices,
        and diodes. Should be called only before passing to write function, as
        reshaped images cannot be extended with other image data.

        Parameters
        ----------
        diodes : int
            Number of diodes of current instrument.
        """
        t0 = time.time()
        image, max_slices = list_to_array(self.image, diodes)
        print(time.time() - t0)
        t0 = time.time()
        #self.image = numpy.reshape(image, (-1, max_slices // diodes, diodes))
        self.image = _reshape_image(image, max_slices, diodes)
        print(time.time() - t0)

    def conv_to_array(self, diodes):
        """ Converts list of 2-D arrays to 1-D numpy array. Should be called
        only before passing to write function, as reshaped images cannot be
        extended with other image data.

        Parameters
        ----------
        diodes : int
            Number of diodes of current instrument.
        """
        # self.image = list(itertools.chain.from_iterable(self.image))
        flatten = numpy.concatenate(self.image)
        self.image = flatten.ravel()

