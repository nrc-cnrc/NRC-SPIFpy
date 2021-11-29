import datetime

import numpy as np
from scipy.interpolate import interp1d

from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.input.spec.buffer import Buffer
from nrc_spifpy.input.spec.image import AssembledImageRecordContainer
from nrc_spifpy.input.spec.housekeeping import add_housekeeping_variables, housekeeping_template, create_housekeeping_group

from nrc_spifpy.input.spec.image import add_auxiliary_core_variables

from tqdm import tqdm

import joblib as jb

file_buffer_template = np.dtype([
    ('year', 'u2'),
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
        self.file_dtype = file_buffer_template
        self.aux_channels = None
        self.chunk_size = 25


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

        self.timestamps = [(x - self.start_date).total_seconds() for x in self.datetimes]

    def process_file(self, spiffile, processors):

        spiffile.set_start_date(self.start_date.strftime('%Y-%m-%d %H:%M:%S %z'))

        group_no_channel = self.name
        group_h_channel = self.name + '-H'
        group_v_channel = self.name + '-V'

        if self.name == '2DS':
            spiffile.create_inst_group(group_h_channel)
            spiffile.create_inst_group(group_v_channel)
            spiffile.set_filenames_attr(group_h_channel, self.filename)
            spiffile.set_filenames_attr(group_v_channel, self.filename)

            add_auxiliary_core_variables(spiffile, group_h_channel)
            add_auxiliary_core_variables(spiffile, group_v_channel)

            create_housekeeping_group(spiffile, group_h_channel)
            create_housekeeping_group(spiffile, group_v_channel)

            add_housekeeping_variables(spiffile, group_h_channel, len(self.data))
            add_housekeeping_variables(spiffile, group_v_channel, len(self.data))

        else:
            spiffile.create_inst_group(group_no_channel)
            spiffile.set_filenames_attr(group_no_channel, self.filename)

            create_housekeeping_group(spiffile, group_no_channel)

            add_auxiliary_core_variables(spiffile, group_no_channel)
            add_housekeeping_variables(spiffile, group_no_channel, len(self.data))


        spiffile.write_buffer_info(self.start_date, self.datetimes)

        pbar1 = tqdm(desc='Processing frames',
                     total=len(self.data),
                     unit='frame')

        chunk_intervals = self.get_chunk_intervals()

        for i in range(len(chunk_intervals) - 1):
            chunk_slice = slice(chunk_intervals[i], chunk_intervals[i + 1])
            
            buffer_ids = [x for x in range(chunk_intervals[i], chunk_intervals[i + 1])]
            buffer_timestamps = self.timestamps[chunk_slice]
            buffers = self.data[chunk_slice]['data']

            chunk_processor = ChunkedBufferProcessor(buffer_ids, buffer_timestamps, buffers)
            chunk_processor.process_buffers()
            housekeeping = chunk_processor.merge_housekeeping()
            images_h, images_v = chunk_processor.merge_assembled_images()

            if self.name == '2DS':
                self.write_image_data(spiffile, group_h_channel, images_h)
                self.write_image_data(spiffile, group_v_channel, images_v)

                self.write_housekeeping_data(spiffile, group_h_channel, housekeeping)
                self.write_housekeeping_data(spiffile, group_v_channel, housekeeping)

            else:
                self.write_image_data(spiffile, group_no_channel, images_v)
                self.write_housekeeping_data(spiffile, group_no_channel, housekeeping)

            pbar1.update(chunk_intervals[i + 1] - chunk_intervals[i])

        tas_post_processor = SPECFileTASPostProcessor()

        if self.name == '2DS':
            tas_post_processor.post_process_tas(spiffile.rootgrp[group_h_channel])
            tas_post_processor.post_process_tas(spiffile.rootgrp[group_v_channel])
        else:
            tas_post_processor.post_process_tas(spiffile.rootgrp[group_no_channel])

    def get_chunk_intervals(self):
        data_len = len(self.data)

        chunk_intervals = np.arange(0, data_len, self.chunk_size)

        if chunk_intervals[-1] != data_len:
            chunk_intervals[-1] = data_len

        return chunk_intervals


    def write_image_data(self, spiffile, instrument, images):

        """ Called each time number of unsaved processed images exceeds
        10,000. Writes H and V images to SPIF NetCDF file.
        Parameters
        ----------
        spiffile : SPIFFile object
            SPIFFile object of current SPIF NetCDF output file
        h_p : Images object
            Images object containing processed H array image info.
        v_p : Images object
            Images object containing processed V array image info.
        """

        instgrp = spiffile.rootgrp[instrument]
        coregrp = spiffile.rootgrp[instrument]['core']

        dim_size = len(instgrp.dimensions['Images'])
        px_size = len(instgrp.dimensions['Pixels'])

        # Primary, critical SPIF variables

        spiffile.write_variable(coregrp, 'image_sec', images.image_sec, dim_size)
        spiffile.write_variable(coregrp, 'image_ns', images.image_ns, dim_size)
        spiffile.write_variable(coregrp, 'image_len', images.image_len, dim_size)
        spiffile.write_variable(coregrp, 'buffer_index', images.buffer_id, dim_size)
        spiffile.write_variable(coregrp, 'image', images.image, px_size)

        # Auxiliary SPIF variables

        spiffile.write_variable(coregrp, 'num_words', images.num_words, dim_size)
        spiffile.write_variable(coregrp, 'timing_flag', images.timing_flag, dim_size)
        spiffile.write_variable(coregrp, 'mismatch_flag', images.mismatch_flag, dim_size)
        spiffile.write_variable(coregrp, 'fifo_flag', images.fifo_flag, dim_size)
        spiffile.write_variable(coregrp, 'overload_flag', images.overload_flag, dim_size)
        spiffile.write_variable(coregrp, 'particle_count', images.particle_count, dim_size)
        spiffile.write_variable(coregrp, 'num_slices', images.num_slices, dim_size)
        spiffile.write_variable(coregrp, 'timeword_upper', images.timeword_upper, dim_size)
        spiffile.write_variable(coregrp, 'timeword_lower', images.timeword_lower, dim_size)
        spiffile.write_variable(coregrp, 'tas', images.tas, dim_size)

        spiffile.rootgrp.sync()

    def write_housekeeping_data(self, spiffile, instrument, housekeeping):
        """ 
        Called each time number of unsaved processed images exceeds
        10,000. Writes H and V images to SPIF NetCDF file.
        Parameters
        ----------
        spiffile : SPIFFile object
            SPIFFile object of current SPIF NetCDF output file
        h_p : Images object
            Images object containing processed H array image info.
        v_p : Images object
            Images object containing processed V array image info.
        """

        instgrp = spiffile.rootgrp[instrument]
        hk_group = spiffile.rootgrp[instrument]['aux-housekeeping']

        dim_size = len(hk_group.dimensions['Packet'])

        # Primary, critical SPIF variables

        spiffile.write_variable( hk_group, "buffer_index",                  housekeeping["buffer_id"][:],                     dim_size )
        spiffile.write_variable( hk_group, "buffer_sec",                    housekeeping["buffer_sec"][:],                    dim_size )
        spiffile.write_variable( hk_group, "buffer_ns",                     housekeeping["buffer_ns"][:],                     dim_size )
        spiffile.write_variable( hk_group, "incomplete_packet",             housekeeping["incomplete_packet"][:],             dim_size )
        spiffile.write_variable( hk_group, "hz_elem_0_voltage",             housekeeping["hz_elem_0_voltage"][:],             dim_size )
        spiffile.write_variable( hk_group, "hz_elem_64_voltage",            housekeeping["hz_elem_64_voltage"][:],            dim_size )
        spiffile.write_variable( hk_group, "hz_elem_127_voltage",           housekeeping["hz_elem_127_voltage"][:],           dim_size )
        spiffile.write_variable( hk_group, "v_elem_0_voltage",              housekeeping["v_elem_0_voltage"][:],              dim_size )
        spiffile.write_variable( hk_group, "v_elem_64_voltage",             housekeeping["v_elem_64_voltage"][:],             dim_size )
        spiffile.write_variable( hk_group, "v_elem_127_voltage",            housekeeping["v_elem_127_voltage"][:],            dim_size )
        spiffile.write_variable( hk_group, "raw_pos_power_supply",          housekeeping["raw_pos_power_supply"][:],          dim_size )
        spiffile.write_variable( hk_group, "raw_neg_power_supply",          housekeeping["raw_neg_power_supply"][:],          dim_size )
        spiffile.write_variable( hk_group, "hz_arm_tx_temp",                housekeeping["hz_arm_tx_temp"][:],                dim_size )
        spiffile.write_variable( hk_group, "hz_arm_rx_temp",                housekeeping["hz_arm_rx_temp"][:],                dim_size )
        spiffile.write_variable( hk_group, "v_arm_tx_temp",                 housekeeping["v_arm_tx_temp"][:],                 dim_size )
        spiffile.write_variable( hk_group, "v_arm_rx_temp",                 housekeeping["v_arm_rx_temp"][:],                 dim_size )
        spiffile.write_variable( hk_group, "hz_tip_tx_temp",                housekeeping["hz_tip_tx_temp"][:],                dim_size )
        spiffile.write_variable( hk_group, "hz_tip_rx_temp",                housekeeping["hz_tip_rx_temp"][:],                dim_size )
        spiffile.write_variable( hk_group, "rear_opt_bridge_temp",          housekeeping["rear_opt_bridge_temp"][:],          dim_size )
        spiffile.write_variable( hk_group, "dsp_board_temp",                housekeeping["dsp_board_temp"][:],                dim_size )
        spiffile.write_variable( hk_group, "forward_vessel_temp",           housekeeping["forward_vessel_temp"][:],           dim_size )
        spiffile.write_variable( hk_group, "hz_laser_temp",                 housekeeping["hz_laser_temp"][:],                 dim_size )
        spiffile.write_variable( hk_group, "v_laser_temp",                  housekeeping["v_laser_temp"][:],                  dim_size )
        spiffile.write_variable( hk_group, "front_plate_temp",              housekeeping["front_plate_temp"][:],              dim_size )
        spiffile.write_variable( hk_group, "power_supply_temp",             housekeeping["power_supply_temp"][:],             dim_size )
        spiffile.write_variable( hk_group, "minus_5V_supply",               housekeeping["minus_5V_supply"][:],               dim_size )
        spiffile.write_variable( hk_group, "plus_5V_supply",                housekeeping["plus_5V_supply"][:],                dim_size )
        spiffile.write_variable( hk_group, "can_internal_pressure",         housekeeping["can_internal_pressure"][:],         dim_size )      
        spiffile.write_variable( hk_group, "hz_elem_21_voltage",            housekeeping["hz_elem_21_voltage"][:],            dim_size )
        spiffile.write_variable( hk_group, "hz_elem_42_voltage",            housekeeping["hz_elem_42_voltage"][:],            dim_size )
        spiffile.write_variable( hk_group, "hz_elem_85_voltage",            housekeeping["hz_elem_85_voltage"][:],            dim_size )
        spiffile.write_variable( hk_group, "hz_elem_106_voltage",           housekeeping["hz_elem_106_voltage"][:],           dim_size )
        spiffile.write_variable( hk_group, "v_elem_21_voltage",             housekeeping["v_elem_21_voltage"][:],             dim_size )
        spiffile.write_variable( hk_group, "v_elem_42_voltage",             housekeeping["v_elem_42_voltage"][:],             dim_size )
        spiffile.write_variable( hk_group, "v_elem_85_voltage",             housekeeping["v_elem_85_voltage"][:],             dim_size )
        spiffile.write_variable( hk_group, "v_elem_106_voltage",            housekeeping["v_elem_106_voltage"][:],            dim_size )
        spiffile.write_variable( hk_group, "num_v_particles_detected",      housekeeping["num_v_particles_detected"][:],      dim_size )
        spiffile.write_variable( hk_group, "num_h_particles_detected",      housekeeping["num_h_particles_detected"][:],      dim_size )
        spiffile.write_variable( hk_group, "heater_outputs",                housekeeping["heater_outputs"][:],                dim_size )
        spiffile.write_variable( hk_group, "h_laser_drive",                 housekeeping["h_laser_drive"][:],                 dim_size )
        spiffile.write_variable( hk_group, "v_laser_drive",                 housekeeping["v_laser_drive"][:],                 dim_size )
        spiffile.write_variable( hk_group, "hz_masked_bits",                housekeeping["hz_masked_bits"][:],                dim_size )
        spiffile.write_variable( hk_group, "v_masked_bits",                 housekeeping["v_masked_bits"][:],                 dim_size )
        spiffile.write_variable( hk_group, "num_stereo_particles_detected", housekeeping["num_stereo_particles_detected"][:], dim_size )
        spiffile.write_variable( hk_group, "num_t_word_mismatch",           housekeeping["num_t_word_mismatch"][:],           dim_size )
        spiffile.write_variable( hk_group, "num_slice_count_mismatch",      housekeeping["num_slice_count_mismatch"][:],      dim_size )
        spiffile.write_variable( hk_group, "num_hz_ov_periods",             housekeeping["num_hz_ov_periods"][:],             dim_size )
        spiffile.write_variable( hk_group, "num_v_ov_periods",              housekeeping["num_v_ov_periods"][:],              dim_size )
        spiffile.write_variable( hk_group, "compression_conf",              housekeeping["compression_conf"][:],              dim_size )
        spiffile.write_variable( hk_group, "num_empty_fifo",                housekeeping["num_empty_fifo"][:],                dim_size )
        spiffile.write_variable( hk_group, "tas",                           housekeeping["tas"][:],                           dim_size )
        spiffile.write_variable( hk_group, "timing_word_1",                 housekeeping["timing_word_1"][:],                 dim_size )
        spiffile.write_variable( hk_group, "timing_word_2",                 housekeeping["timing_word_2"][:],                 dim_size )

        spiffile.rootgrp.sync()

class ChunkedBufferProcessor:

    def __init__(self, buffer_ids, buffer_timestamps, raw_buffers):
        self.buffer_ids = buffer_ids
        self.buffer_timestamps = buffer_timestamps
        self.raw_buffers = raw_buffers

        self.processed_buffers = [None]*len(raw_buffers)

        self.output = [None]*len(raw_buffers)


    def process_buffers(self):

        for i in range(len(self.output)):
            self.output[i] = self.process_buffer(
                self.buffer_ids[i],
                self.buffer_timestamps[i],
                self.raw_buffers[i],
                i
            )

    def process_buffer(self, buffer_id, buffer_timestamp, buffer, output_id):

        buffer = Buffer(buffer_id, buffer_timestamp, buffer)
        self.processed_buffers[output_id] = buffer

        return buffer.assembled_images

    def merge_assembled_images(self):

        image_container_h = AssembledImageRecordContainer()
        image_container_v = AssembledImageRecordContainer()

        for k in image_container_h.__dict__.keys():
            image_container_h.__dict__[k] = np.concatenate([x['h'].__dict__[k] for x in self.output])
            image_container_v.__dict__[k] = np.concatenate([x['v'].__dict__[k] for x in self.output])

        return image_container_h, image_container_v

    def merge_housekeeping(self):
        housekeeping_data = np.array([], dtype = housekeeping_template)

        for buffer in self.processed_buffers:
            housekeeping_data = np.append(housekeeping_data, buffer.housekeeping)

        housekeeping_data['housekeeping_packet_id'] = np.arange(0,len(housekeeping_data))

        return housekeeping_data

class SPECFileTASPostProcessor:

    def __init__(self) -> None:
        pass

    def post_process_tas(self, instfile):

        aux_hk_group = instfile["aux-housekeeping"]
        core_group = instfile["core"]

        self.clean_hk_tas(aux_hk_group)
        self.set_core_tas(aux_hk_group, core_group)

    def clean_hk_tas(self, aux_hk_group):
        tas_clean = aux_hk_group['tas'][:]

        first_clean_tas_index = np.argwhere( ~np.isnan(tas_clean) )[0]

        # do back filling first just in case

        if first_clean_tas_index != 0:
            tas_clean[0:first_clean_tas_index] = tas_clean[first_clean_tas_index]

        for i in range(1,len(tas_clean)):
            if np.isnan(tas_clean[i]):
                tas_clean[i] = tas_clean[i - 1]

        aux_hk_group['tas'][:] = tas_clean

    def set_core_tas(self, aux_hk_group, core_group):
        buffer_id_hk = aux_hk_group['buffer_index'][:]
        tas_hk = aux_hk_group['tas'][:]

        buffer_id_core = core_group['buffer_index'][:]
        tas_core = core_group['tas'][:]

        f = interp1d(buffer_id_hk, tas_hk, kind = 'previous', fill_value='extrapolate')

        interpolated_tas = f( np.arange(0,np.max(buffer_id_core) + 1) )

        updated_core_tas = interpolated_tas[buffer_id_core]

        core_group['tas'][:] = updated_core_tas


        