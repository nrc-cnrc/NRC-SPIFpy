import configparser
import os
import tempfile
import unittest

import numpy

from nrc_spifpy.images import Images
from nrc_spifpy.input.dmt_mono_file import DMTMonoFile
from nrc_spifpy.spif import SPIFFile


class TestDMTMonoMetadata(unittest.TestCase):
    def test_netcdf_preserves_dof_flag_dtype_values_and_attributes(self):
        dmt = DMTMonoFile('dummy.CIP', 'CIP', 15)
        images = Images(dmt.aux_channels)
        for index, dof_flag in enumerate((0, 1)):
            images.sec.append(0)
            images.ns.append(index)
            images.image.append(numpy.ones(64, dtype=numpy.uint8))
            images.length.append(1)
            images.buffer_index.append(0)
            images.image_count.append((65535 + index) % (1 << 16))
            images.dof_flag.append(dof_flag)
        images.conv_to_array(dmt.diodes)

        config = configparser.ConfigParser()
        config.read(os.path.join(
            os.path.dirname(__file__),
            '..',
            'nrc_spifpy',
            'config',
            'CIP.ini',
        ))

        with tempfile.TemporaryDirectory() as temp_dir:
            output = SPIFFile(os.path.join(temp_dir, 'dmt.nc'), config)
            output.create_file()
            try:
                output.create_inst_group('CIP')
                dmt._write_images(output, images)
                core = output.instgrps['CIP']['core']
                dof = core['dof_flag']

                self.assertEqual(dof.dtype, numpy.dtype('uint8'))
                numpy.testing.assert_array_equal(dof[:], [0, 1])
                self.assertEqual(int(dof._FillValue), 255)
                self.assertEqual(dof.long_name, 'DMT depth-of-field flag')
                self.assertEqual(dof.units, '1')
                numpy.testing.assert_array_equal(dof.flag_values, [0, 1])
                self.assertEqual(
                    dof.flag_meanings,
                    'out_of_focus meets_depth_of_field_requirement',
                )
                self.assertEqual(
                    dof.description,
                    'Active-high DMT probe flag: 1 indicates that the particle '
                    'meets the depth-of-field requirement for sizing; 0 '
                    'indicates rejection. Missing flags are stored as the fill '
                    'value.',
                )

                counts = core['image_count']
                self.assertEqual(counts.dtype, numpy.dtype('uint16'))
                numpy.testing.assert_array_equal(counts[:], [65535, 0])
                self.assertEqual(
                    counts.long_name, 'DMT particle image counter'
                )
                self.assertEqual(counts.units, '1')
                self.assertEqual(
                    counts.comment,
                    '16-bit counter stored modulo 2^16; discontinuities can '
                    'indicate dropped particles.',
                )
            finally:
                output.close()


if __name__ == '__main__':
    unittest.main()
