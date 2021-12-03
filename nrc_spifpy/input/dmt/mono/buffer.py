import numpy as np
from numba import njit

class ImageBufferDecompressor:

    def __init__(self) -> None:
        pass

    def decompress_buffer(self, buffer):
        decompressed_buffer = decompress_dmt_mono_buffer(buffer)

        return decompressed_buffer
@njit
def decompress_dmt_mono_buffer(data):
    
    decompressed_data = np.array([np.uint8(x) for x in range(0)], dtype = 'B')
    
    i = 0
    while i < len(data):
        rlehb_byte = data[i]
        
        all_zeros = rlehb_all_zeros(rlehb_byte)
        all_ones = rlehb_all_ones(rlehb_byte)
        all_dummy = rlehb_all_dummy(rlehb_byte)
        data_count = rlehb_count(rlehb_byte)
        
        i += 1

        if all_zeros:
            decompressed_data = np.append(decompressed_data, np.zeros(data_count, dtype = 'B'))
        elif all_ones:
            decompressed_data = np.append(decompressed_data, np.zeros(data_count, dtype = 'B') + np.uint8(255) )
        elif all_dummy:
            pass
        else:
            decompressed_data = np.append( decompressed_data, data[i:i + data_count])
            i += data_count

    return decompressed_data

"""
    Functions related to processing Run-Length Encoding Header Bytes(RHELBs)
"""

@njit
def rlehb_all_zeros(rlehb_byte):
    return (rlehb_byte & 0b10000000) >> 7

@njit
def rlehb_all_ones(rlehb_byte):
    return (rlehb_byte & 0b01000000) >> 6

@njit
def rlehb_all_dummy(rlehb_byte):
    return (rlehb_byte & 0b00100000) >> 5

@njit
def rlehb_count(rlehb_byte):
    return (rlehb_byte & 0b00011111) + 1
