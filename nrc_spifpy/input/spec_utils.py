#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
from dataclasses import field

import numpy


@dataclass
class HkElem:
    ident: str
    name: str
    units: str
    dtype: str
    coeffs: list[float] = field(default_factory=list)
    data: list[float] = field(default_factory=list)


def gen_hk_elems():

    volt_coeffs = [0, 0.00244140625]
    temp_coeffs = [1.6, 0.0244140625]
    power_supply_coeffs = [0, 0.00488400488]
    pressure_coeffs = [-3.846, 0.018356]
    no_coeff = [0, 1]
    laser_drive_coeffs = [0, 0.001220703]

    hk_elems = []
    hk_elems.append(HkElem('h_elem_0_voltage',
                           'Horizontal Element 0 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('h_elem_64_voltage',
                           'Horizontal Element 64 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('h_elem_127_voltage',
                           'Horizontal Element 127 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    

    hk_elems.append(HkElem('v_elem_0_voltage',
                           'Vertical Element 0 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('v_elem_64_voltage',
                           'Vertical Element 64 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('v_elem_127_voltage',
                           'Vertical Element 127 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))

    hk_elems.append(HkElem('raw_pos_power_supply',
                           'Raw positive power supply',
                           'volts',
                           'f',
                           power_supply_coeffs))

    hk_elems.append(HkElem('raw_neg_power_supply',
                           'Raw negative power supply',
                           'volts',
                           'f',
                           power_supply_coeffs))

    hk_elems.append(HkElem('h_arm_tx_temp',
                           'Horizonal arm Tx temp',
                           'deg C',
                           'f',
                           temp_coeffs))
    hk_elems.append(HkElem('h_arm_rx_temp',
                           'Horizonal arm Rx temp',
                           'deg C',
                           'f',
                           temp_coeffs))

    hk_elems.append(HkElem('v_arm_tx_temp',
                           'Vertical arm Tx temp',
                           'deg C',
                           'f',
                           temp_coeffs))
    hk_elems.append(HkElem('v_arm_rx_temp',
                           'Vertical arm Rx temp',
                           'deg C',
                           'f',
                           temp_coeffs))

    hk_elems.append(HkElem('h_tip_tx_temp',
                           'Horizonal tip Tx temp',
                           'deg C',
                           'f',
                           temp_coeffs))
    hk_elems.append(HkElem('h_tip_rx_temp',
                           'Horizonal tip Rx temp',
                           'deg C',
                           'f',
                           temp_coeffs))

    hk_elems.append(HkElem('rear_opt_bridge_temp',
                           'Rear optical bridge temp',
                           'deg C',
                           'f',
                           temp_coeffs))
    hk_elems.append(HkElem('dsp_board_temp',
                           'DSP board temp',
                           'deg C',
                           'f',
                           temp_coeffs))
    hk_elems.append(HkElem('forward_vessel_temp',
                           'Forward vessel temp',
                           'deg C',
                           'f',
                           temp_coeffs))

    hk_elems.append(HkElem('h_laser_temp',
                           'Horizonal laser temp',
                           'deg C',
                           'f',
                           temp_coeffs))
    hk_elems.append(HkElem('v_laser_temp',
                           'Vertical laser temp',
                           'deg C',
                           'f',
                           temp_coeffs))

    hk_elems.append(HkElem('front_plate_temp',
                           'Front plate temp',
                           'deg C',
                           'f',
                           temp_coeffs))
    hk_elems.append(HkElem('power_supply_temp',
                           'Power supply temp',
                           'deg C',
                           'f',
                           temp_coeffs))

    hk_elems.append(HkElem('5v_neg_supply',
                           '-5V supply',
                           'volts',
                           'f',
                           power_supply_coeffs))
    hk_elems.append(HkElem('5v_pos_supply',
                           '+5V supply',
                           'volts',
                           'f',
                           power_supply_coeffs))

    hk_elems.append(HkElem('can_internal_pressure',
                           'Canister internal pressure',
                           'psi',
                           'f',
                           pressure_coeffs))

    hk_elems.append(HkElem('h_elem_21_voltage',
                           'Horizontal Element 21 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('h_elem_42_voltage',
                           'Horizontal Element 42 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('h_elem_85_voltage',
                           'Horizontal Element 85 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('h_elem_106_voltage',
                           'Horizontal Element 106 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))

    hk_elems.append(HkElem('v_elem_21_voltage',
                           'Vertical Element 21 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('v_elem_42_voltage',
                           'Vertical Element 42 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('v_elem_85_voltage',
                           'Vertical Element 85 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))
    hk_elems.append(HkElem('v_elem_106_voltage',
                           'Vertical Element 106 Voltage',
                           'volts',
                           'f',
                           volt_coeffs))

    hk_elems.append(HkElem('v_particles_detected',
                           'Vertical particles detected',
                           'counts',
                           'f',
                           no_coeff))
    hk_elems.append(HkElem('h_particles_detected',
                           'Horizontal particles detected',
                           'counts',
                           'f',
                           no_coeff))

    hk_elems.append(HkElem('heater_outputs',
                           'Heater outputs (1 if on), bit # indicates zone #',
                           'bit map',
                           'f',
                           no_coeff))

    hk_elems.append(HkElem('h_laser_drive',
                           'Horizontal laser drive',
                           'volts',
                           'f',
                           laser_drive_coeffs))
    hk_elems.append(HkElem('v_laser_drive',
                           'Vertical laser drive',
                           'volts',
                           'f',
                           laser_drive_coeffs))

    hk_elems.append(HkElem('h_masked_bits',
                           'Horizontal masked bits',
                           'counts',
                           'f',
                           no_coeff))
    hk_elems.append(HkElem('v_masked_bits',
                           'Vertical masked bits',
                           'counts',
                           'f',
                           no_coeff))

    hk_elems.append(HkElem('n_stereo',
                           'Number of stereo particles found',
                           'counts',
                           'u2',
                           no_coeff))
    hk_elems.append(HkElem('n_timing_mismatch',
                           'Number of timing word mismatches',
                           'counts',
                           'u2',
                           no_coeff))
    hk_elems.append(HkElem('n_slice_mismatch',
                           'Number of slice count mismatches',
                           'counts',
                           'u2',
                           no_coeff))
    hk_elems.append(HkElem('h_overload',
                           'Number of horizontal overload periods',
                           'counts',
                           'u2',
                           no_coeff))
    hk_elems.append(HkElem('v_overload',
                           'Number of vertical overload periods',
                           'counts',
                           'u2',
                           no_coeff))

    hk_elems.append(HkElem('compression_config',
                           'Compression configuration',
                           'counts',
                           'f',
                           no_coeff))

    hk_elems.append(HkElem('n_empty_fifo',
                           'Number of empty FIFO faults',
                           'counts',
                           'u2',
                           no_coeff))

    return hk_elems

def process_spec_aux(spiffile, inst_name, data, datetimes, start_date):
    hk_dim = 'Packet'

    hk_group = spiffile.rootgrp[inst_name].createGroup('aux-housekeeping')
    hk_group.createDimension(hk_dim, None)

    hk_elems = gen_hk_elems()

    frame_ns, frame_secs, frame_indx, tas_vals, hk_elems = extract_hk(data, hk_elems, datetimes, start_date)

    hk_elems.append(HkElem('buffer_sec',
                           'buffer time in seconds',
                           'seconds since start_date',
                           'u4',
                           data=frame_secs))

    hk_elems.append(HkElem('buffer_ns',
                           'buffer time in nanoseconds',
                           'nanoseconds since buffer_sec',
                           'u4',
                           data=frame_ns))

    hk_elems.append(HkElem('buffer_index',
                           'buffer index',
                           '',
                           'u4',
                           data=frame_indx))

    hk_elems.append(HkElem('tas',
                           'true airspeed',
                           'm/s',
                           'f',
                           data=tas_vals))

    for elem in hk_elems:
        elem_attrs = {'long_name': elem.name,
                      'units': elem.units}
        spiffile.create_variable(hk_group,
                                 elem.ident,
                                 elem.dtype,
                                 (hk_dim,),
                                 attrs=elem_attrs,
                                 data=elem.data
                                 )


def extract_hk(data, hk_elems, datetimes, start_date):
    times = numpy.array(datetimes, dtype='datetime64[ns]') - numpy.datetime64(start_date)
    secs = times.astype('timedelta64[s]')
    ns = times - secs
    frame_secs = []
    frame_ns = []
    frame_indx = []
    tas_vals = []

    for i, frame in enumerate(data):
        record = frame['data']
        hk_indx = numpy.where(record == 18507)[0]

        for indx in hk_indx:
            try:
                tas = numpy.array([record[indx + 50],
                                   record[indx + 49]],
                                  dtype='u2').view('float32')[0]
                tas_dec = tas % 1
                if tas > 0 and tas < 1000 and tas_dec == 0: # check for good data
                    tas_vals.append(tas)
                    for j, elem in enumerate(hk_elems):
                        elem.data.append(record[indx + j + 1] * elem.coeffs[1] + elem.coeffs[0])
                    frame_secs.append(secs[i])
                    frame_ns.append(ns[i])
                    frame_indx.append(i)
            except IndexError:
                # print('here')
                pass

    return frame_ns, frame_secs, frame_indx, tas_vals, hk_elems