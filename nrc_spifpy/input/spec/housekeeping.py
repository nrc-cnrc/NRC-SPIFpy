import numpy as np

housekeeping_template = np.dtype([
    ("housekeeping_packet_id", "i"), # This is to be filled just prior to writing into file
    ("buffer_id","i"),
    ("buffer_sec", "i"),
    ("buffer_ns", "i"),
    ("buffer_index","u2"),
    ("incomplete_packet", "u2"),
    ("hz_elem_0_voltage","f"),
    ("hz_elem_64_voltage","f"),
    ("hz_elem_127_voltage","f"),
    ("v_elem_0_voltage","f"),
    ("v_elem_64_voltage","f"),
    ("v_elem_127_voltage","f"),
    ("raw_pos_power_supply","f"),
    ("raw_neg_power_supply","f"),
    ("hz_arm_tx_temp","f"),
    ("hz_arm_rx_temp","f"),
    ("v_arm_tx_temp","f"),
    ("v_arm_rx_temp","f"),
    ("hz_tip_tx_temp","f"),
    ("hz_tip_rx_temp","f"),
    ("rear_opt_bridge_temp","f"),
    ("dsp_board_temp","f"),
    ("forward_vessel_temp","f"),
    ("hz_laser_temp","f"),
    ("v_laser_temp","f"),
    ("front_plate_temp","f"),
    ("power_supply_temp","f"),
    ("minus_5V_supply","f"),
    ("plus_5V_supply","f"),
    ("can_internal_pressure","f"),
    ("hz_elem_21_voltage","f"),
    ("hz_elem_42_voltage","f"),
    ("hz_elem_85_voltage","f"),
    ("hz_elem_106_voltage","f"),
    ("v_elem_21_voltage","f"),
    ("v_elem_42_voltage","f"),
    ("v_elem_85_voltage","f"),
    ("v_elem_106_voltage","f"),
    ("num_v_particles_detected","u2"),
    ("num_h_particles_detected","u2"),
    ("heater_outputs","u2"),
    ("h_laser_drive","f"),
    ("v_laser_drive","f"),
    ("hz_masked_bits","u2"),
    ("v_masked_bits","u2"),
    ("num_stereo_particles_detected","u2"),
    ("num_t_word_mismatch","u2"),
    ("num_slice_count_mismatch","u2"),
    ("num_hz_ov_periods","u2"),
    ("num_v_ov_periods","u2"),
    ("compression_conf","u2"),
    ("num_empty_fifo","u2"),
    ("spare_2","u2"),
    ("spare_3","u2"),
    ("tas","f"),
    ("timing_word_1","u2"),
    ("timing_word_2","u2")
])

def process_housekeeping(buffer_id, buffer_sec, buffer_ns, buffer_index, raw_housekeeping):
    housekeeping_container = np.zeros(1, dtype = housekeeping_template)

    housekeeping_container['buffer_id'][:] = buffer_id
    housekeeping_container['buffer_sec'][:] = buffer_sec
    housekeeping_container['buffer_ns'][:] = buffer_ns
    housekeeping_container['buffer_index'][:] = buffer_index

    if len(raw_housekeeping) < 53:
        raw_housekeeping = np.append( raw_housekeeping, [0] * (53 - len(raw_housekeeping)) )
        housekeeping_container['incomplete_packet'][:] = 1
    
    housekeeping_container["hz_elem_0_voltage"][:] =                 raw_housekeeping[1]  * 0.00244140625
    housekeeping_container["hz_elem_64_voltage"][:] =                raw_housekeeping[2]  * 0.00244140625
    housekeeping_container["hz_elem_127_voltage"][:] =               raw_housekeeping[3]  * 0.00244140625
    housekeeping_container["v_elem_0_voltage"][:] =                  raw_housekeeping[4]  * 0.00244140625
    housekeeping_container["v_elem_64_voltage"][:] =                 raw_housekeeping[5]  * 0.00244140625
    housekeeping_container["v_elem_127_voltage"][:] =                raw_housekeeping[6]  * 0.00244140625
    housekeeping_container["raw_pos_power_supply"][:] =              raw_housekeeping[7]  * 0.00488400488
    housekeeping_container["raw_neg_power_supply"][:] =              raw_housekeeping[8]  * 0.00488400488
    housekeeping_container["hz_arm_tx_temp"][:] =                    raw_housekeeping[9]  * 0.00244140625 + 1.6
    housekeeping_container["hz_arm_rx_temp"][:] =                    raw_housekeeping[10] * 0.00244140625 + 1.6
    housekeeping_container["v_arm_tx_temp"][:] =                     raw_housekeeping[11] * 0.00244140625 + 1.6
    housekeeping_container["v_arm_rx_temp"][:] =                     raw_housekeeping[12] * 0.00244140625 + 1.6
    housekeeping_container["hz_tip_tx_temp"][:] =                    raw_housekeeping[13] * 0.00244140625 + 1.6
    housekeeping_container["hz_tip_rx_temp"][:] =                    raw_housekeeping[14] * 0.00244140625 + 1.6
    housekeeping_container["rear_opt_bridge_temp"][:] =              raw_housekeeping[15] * 0.00244140625 + 1.6
    housekeeping_container["dsp_board_temp"][:] =                    raw_housekeeping[16] * 0.00244140625 + 1.6
    housekeeping_container["forward_vessel_temp"][:] =               raw_housekeeping[17] * 0.00244140625 + 1.6
    housekeeping_container["hz_laser_temp"][:] =                     raw_housekeeping[18] * 0.00244140625 + 1.6
    housekeeping_container["v_laser_temp"][:] =                      raw_housekeeping[19] * 0.00244140625 + 1.6
    housekeeping_container["front_plate_temp"][:] =                  raw_housekeeping[20] * 0.00244140625 + 1.6
    housekeeping_container["power_supply_temp"][:] =                 raw_housekeeping[21] * 0.00244140625 + 1.6
    housekeeping_container["minus_5V_supply"][:] =                   raw_housekeeping[22] * 0.00488400488
    housekeeping_container["plus_5V_supply"][:] =                    raw_housekeeping[23] * 0.00488400488
    housekeeping_container["can_internal_pressure"][:] =             raw_housekeeping[24] * 0.01835600000 - 3.846
    housekeeping_container["hz_elem_21_voltage"][:] =                raw_housekeeping[25] * 0.00244140625
    housekeeping_container["hz_elem_42_voltage"][:] =                raw_housekeeping[26] * 0.00244140625
    housekeeping_container["hz_elem_85_voltage"][:] =                raw_housekeeping[27] * 0.00244140625
    housekeeping_container["hz_elem_106_voltage"][:] =               raw_housekeeping[28] * 0.00244140625
    housekeeping_container["v_elem_21_voltage"][:] =                 raw_housekeeping[29] * 0.00244140625
    housekeeping_container["v_elem_42_voltage"][:] =                 raw_housekeeping[30] * 0.00244140625
    housekeeping_container["v_elem_85_voltage"][:] =                 raw_housekeeping[31] * 0.00244140625
    housekeeping_container["v_elem_106_voltage"][:] =                raw_housekeeping[32] * 0.00244140625
    housekeeping_container["num_v_particles_detected"][:] =          raw_housekeeping[33]
    housekeeping_container["num_h_particles_detected"][:] =          raw_housekeeping[34]
    housekeeping_container["heater_outputs"][:] =                    raw_housekeeping[35]
    housekeeping_container["h_laser_drive"][:] =                     raw_housekeeping[36] * 0.001220703
    housekeeping_container["v_laser_drive"][:] =                     raw_housekeeping[37] * 0.001220703
    housekeeping_container["hz_masked_bits"][:] =                    raw_housekeeping[38]
    housekeeping_container["v_masked_bits"][:] =                     raw_housekeeping[39]
    housekeeping_container["num_stereo_particles_detected"][:] =     raw_housekeeping[40]
    housekeeping_container["num_t_word_mismatch"][:] =               raw_housekeeping[41]
    housekeeping_container["num_slice_count_mismatch"][:] =          raw_housekeeping[42]
    housekeeping_container["num_hz_ov_periods"][:] =                 raw_housekeeping[43]
    housekeeping_container["num_v_ov_periods"][:] =                  raw_housekeeping[44]
    housekeeping_container["compression_conf"][:] =                  raw_housekeeping[45]
    housekeeping_container["num_empty_fifo"][:] =                    raw_housekeeping[46]
    housekeeping_container["tas"][:] =                               np.array(
                                                                        [raw_housekeeping[50],raw_housekeeping[49]], 
                                                                        dtype=np.uint16
                                                                    ).view("f")[0]
    housekeeping_container["timing_word_1"][:] =                     raw_housekeeping[51]
    housekeeping_container["timing_word_2"][:] =                     raw_housekeeping[52]  

    return housekeeping_container

# A utility method to create the housekeeping group in a netcdf file

def create_housekeeping_group(spiffile, inst_name):
    hk_group = spiffile.rootgrp[inst_name].createGroup('aux-housekeeping')
    hk_group.createDimension('Packet', None)

# A utility method to define the extra image related variables in a netcdf file

def add_housekeeping_variables(spiffile, inst_name, num_buffers):
    hk_group = spiffile.rootgrp[inst_name]['aux-housekeeping']

    if num_buffers < 128:
        chunk_size = num_buffers
    else:
        chunk_size = 128

    spiffile.create_variable( hk_group, "buffer_index",                 "i",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "buffer_sec",                   "i",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "buffer_ns",                    "i",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "incomplete_packet",            "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_elem_0_voltage",            "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_elem_64_voltage",           "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_elem_127_voltage",          "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_elem_0_voltage",             "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_elem_64_voltage",            "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_elem_127_voltage",           "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "raw_pos_power_supply",         "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "raw_neg_power_supply",         "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_arm_tx_temp",               "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_arm_rx_temp",               "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_arm_tx_temp",                "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_arm_rx_temp",                "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_tip_tx_temp",               "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_tip_rx_temp",               "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "rear_opt_bridge_temp",         "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "dsp_board_temp",               "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "forward_vessel_temp",          "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_laser_temp",                "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_laser_temp",                 "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "front_plate_temp",             "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "power_supply_temp",            "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "minus_5V_supply",              "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "plus_5V_supply",               "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "can_internal_pressure",        "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_elem_21_voltage",           "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_elem_42_voltage",           "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_elem_85_voltage",           "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_elem_106_voltage",          "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_elem_21_voltage",            "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_elem_42_voltage",            "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_elem_85_voltage",            "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_elem_106_voltage",           "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "num_v_particles_detected",     "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "num_h_particles_detected",     "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "heater_outputs",               "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "h_laser_drive",                "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_laser_drive",                "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "hz_masked_bits",               "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "v_masked_bits",                "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "num_stereo_particles_detected","u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "num_t_word_mismatch",          "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "num_slice_count_mismatch",     "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "num_hz_ov_periods",            "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "num_v_ov_periods",             "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "compression_conf",             "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "num_empty_fifo",               "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "spare_2",                      "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "spare_3",                      "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "tas",                          "f",  ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "timing_word_1",                "u2", ('Packet',), None, chunksizes=(chunk_size,) )
    spiffile.create_variable( hk_group, "timing_word_2",                "u2", ('Packet',), None, chunksizes=(chunk_size,) )