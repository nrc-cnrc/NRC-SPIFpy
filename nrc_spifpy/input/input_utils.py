#!/usr/bin/env python
# coding: utf-8

import numpy

from nrc_spifpy.input import sea


def read_sea_image_data(filename, file_dtype, typ, inst_name=None):

    if typ == 'mono':
        probe_tag = sea.SEA_ACQUISITION_TYPE_CIP_IMAGE
    elif typ == 'grey':
        probe_tag = sea.SEA_ACQUISITION_TYPE_2D_GREY_ADVANCED
    elif typ == '2d':
        probe_tag = sea.SEA_ACQUISITION_TYPE_2D_MONO_IMAGE

    time, dset = read_sea_file(filename, probe_tag, inst_name)

    dset_len = len(dset)
    print(dset_len)

    if dset_len == 0:
        raise ValueError(f'No {inst_name} data present in {filename}.')

    year = [t.year for t in time]
    month = [t.month for t in time]
    day = [t.day for t in time]
    hour = [t.hour for t in time]
    minute = [t.minute for t in time]
    second = [t.second for t in time]
    ms = [int(t.microsecond / 1000) for t in time]

    data = numpy.empty(dset_len, dtype=file_dtype)

    data['year'] = year
    data['month'] = month
    data['day'] = day
    data['hour'] = hour
    data['minute'] = minute
    data['second'] = second
    data['ms'] = ms
    data['data'] = dset

    return data


def read_sea_tas(filename, typ, resolution):

    if typ is '2dc':
        tas_tag = 6  # 2D Mono TAS Factors

    time, dset = read_sea_file(filename, tas_tag)

    tas = []
    for data in dset:
        tas.append(calc_2dc_tas(data, resolution))

    return tas


def calc_2dc_tas(parameters, resolution):

    return 50000 * resolution * parameters[0] / parameters[1] / 1000000


def read_sea_file(filename, tag, inst_name=None):

    seafile = sea.SEAFile(filename)

    tags = seafile.get_tags_by_typ(tag)

    length = None

    if tag == sea.SEA_ACQUISITION_TYPE_CIP_IMAGE:
        length = 4096
    elif tag == sea.SEA_ACQUISITION_TYPE_2D_MONO_IMAGE:
        length = 1024

    if inst_name is not None:
        tags = [t for t in tags if inst_name.lower() in t.description.lower()]

    dset = []
    time = []
    for buf in seafile.iter_buffers({'tag_number': [t.tag_number
                                     for t in tags]}):

        time.append(sea.datetime_from_seatime(
            buf.get_dataset_by_tag_number(sea.TIME_TAG)))

        dset.append(buf.get_datasets_by_typ(tag)[0][:length])

    return time, dset