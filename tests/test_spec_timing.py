import configparser
import datetime
import os
import tempfile
import unittest

import numpy

from nrc_spifpy.images import Images
from nrc_spifpy.input.spec_file import SPECFile
from nrc_spifpy.spif import SPIFFile


class TestSPECTiming(unittest.TestCase):
    def setUp(self):
        self.spec = SPECFile('dummy.2DS', '2DS', 10)

    def test_unwraps_32_bit_rollover_with_modular_subtraction(self):
        modulus = 1 << 32

        elapsed = self.spec._unwrap_counter(
            numpy.array([
                modulus - 20,
                modulus - 10,
                0,
                10,
            ], dtype=numpy.uint64),
            modulus,
        )

        numpy.testing.assert_array_equal(elapsed, [0, 10, 20, 30])

    def test_unwraps_48_bit_rollover_with_modular_subtraction(self):
        modulus = 1 << 48

        elapsed = self.spec._unwrap_counter(
            numpy.array([
                modulus - 20,
                modulus - 10,
                0,
                10,
            ], dtype=numpy.uint64),
            modulus,
        )

        numpy.testing.assert_array_equal(elapsed, [0, 10, 20, 30])

    def test_image_times_remain_continuous_across_rollover(self):
        modulus = 1 << 32
        counts = numpy.array([
            modulus - 15_000_000,
            modulus - 10_000_000,
            modulus - 5_000_000,
            0,
            5_000_000,
        ], dtype=numpy.uint64)

        image_time = self.spec._calculate_image_times(
            counts,
            numpy.full(5, 100.0),
            numpy.array([1.0, 1.0, 1.0, 2.0, 2.0]),
        )

        numpy.testing.assert_allclose(
            image_time, [0.0, 0.5, 1.0, 1.5, 2.0]
        )

    def test_hvps4_counter_timing_uses_50_um_not_ini_resolution(self):
        spec = SPECFile('dummy.HVPS4', 'HVPS4', 150)
        counts = numpy.array([0, 2_000_000], dtype=numpy.uint64)

        image_time = spec._calculate_image_times(
            counts,
            numpy.full(2, 100.0),
            numpy.array([0.0, 1.0]),
        )

        self.assertEqual(spec.resolution, 150)
        numpy.testing.assert_allclose(numpy.diff(image_time), [1.0])

    def test_anchor_windows_limit_tas_drift_without_time_steps(self):
        modulus = 1 << 32
        buffer_time = numpy.arange(700, dtype=numpy.float64)
        counts = (
            modulus - 100_000_000
            + numpy.arange(700, dtype=numpy.uint64) * 10_000_000
        ) % modulus

        image_time = self.spec._calculate_image_times(
            counts,
            numpy.full(700, 99.0),
            buffer_time,
        )

        self.assertTrue(numpy.all(numpy.diff(image_time) > 0))
        self.assertLess(numpy.max(numpy.abs(image_time - buffer_time)), 2.0)

    def test_real_gap_starts_a_new_counter_timing_block(self):
        image_time = self.spec._calculate_image_times(
            numpy.array([
                1_000_000,
                6_000_000,
                500,
                5_000_500,
            ], dtype=numpy.uint64),
            numpy.full(4, 100.0),
            numpy.array([0.5, 0.5, 300.5, 300.5]),
        )

        numpy.testing.assert_allclose(
            image_time, [0.0, 0.5, 300.0, 300.5]
        )

    def test_counter_reset_starts_a_new_timing_block(self):
        image_time = self.spec._calculate_image_times(
            numpy.array([
                1_000_000,
                6_000_000,
                100,
                5_000_100,
            ], dtype=numpy.uint64),
            numpy.full(4, 100.0),
            numpy.array([0.5, 0.5, 1.5, 1.5]),
        )

        numpy.testing.assert_allclose(
            image_time, [0.0, 0.5, 1.0, 1.5]
        )

    def test_missing_tas_falls_back_to_buffer_time(self):
        buffer_time = numpy.array([0.0, 0.0, 1.0])

        image_time = self.spec._calculate_image_times(
            numpy.array([10, 20, 30], dtype=numpy.uint64),
            numpy.full(3, numpy.nan),
            buffer_time,
        )

        numpy.testing.assert_array_equal(image_time, buffer_time)

    def test_partial_write_preserves_counter_and_sets_aux_metadata(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(
            os.path.dirname(__file__),
            '..',
            'nrc_spifpy',
            'config',
            '2DS.ini',
        ))
        images = Images(self.spec.aux_channels)
        images.sec.append(0)
        images.ns.append(0)
        images.image.append(numpy.ones(128, dtype=numpy.uint8))
        images.length.append(1)
        images.buffer_index.append(0)
        images.tas.append(100.0)
        images.clock_counts.append((1 << 48) - 1)
        images.overload_flag.append(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = SPIFFile(os.path.join(temp_dir, 'spec.nc'), config)
            output.create_file()
            try:
                output.create_inst_group('2DS-H')
                self.spec._partial_write(output, images, '-H')
                core = output.instgrps['2DS-H']['core']
                tas = core['tas']
                counts = core['clock_counts']
                overload = core['overload_flag']

                self.assertEqual(
                    tas.long_name, 'True airspeed as recorded by probe'
                )
                self.assertEqual(tas.units, 'm/s')
                self.assertEqual(counts.dtype, numpy.dtype('uint64'))
                self.assertEqual(int(counts[0]), (1 << 48) - 1)
                self.assertEqual(
                    counts.long_name,
                    'Probe clock count at the last image slice',
                )
                self.assertEqual(counts.units, '1')
                self.assertEqual(
                    counts.comment,
                    'Free-running counter stored modulo 2^32 for 2DS and HVPS, '
                    'or modulo 2^48 for HVPS4.',
                )
                self.assertEqual(overload.dtype, numpy.dtype('uint8'))
                self.assertEqual(int(overload[0]), 1)
                self.assertEqual(
                    overload.long_name, 'Probe particle overload status'
                )
                self.assertEqual(overload.units, '1')
                numpy.testing.assert_array_equal(
                    overload.flag_values, [0, 1]
                )
                self.assertEqual(
                    overload.flag_meanings, 'not_overloaded overloaded'
                )
            finally:
                output.close()

    def test_store_image_preserves_decoded_overload_flag(self):
        self.spec.start_date = datetime.datetime(2024, 1, 1)
        images = Images(self.spec.aux_channels)
        flags = self.spec.decode_flags(1 << 15)
        flags['timing'] = 0
        flags['rem'] = 0

        self.spec.store_image(
            flags,
            None,
            numpy.ones(128, dtype=numpy.uint8),
            0,
            images,
            self.spec.start_date,
            0,
            100.0,
            123,
        )

        self.assertEqual(images.overload_flag, [1])

    def test_calc_image_times_writes_normalized_sec_and_ns(self):
        self.spec.start_date = datetime.datetime(2024, 1, 1)
        self.spec.datetimes = numpy.array([
            '2024-01-01T00:00:01',
            '2024-01-01T00:00:01.500',
        ], dtype='datetime64[ns]')

        config = configparser.ConfigParser()
        config.read(os.path.join(
            os.path.dirname(__file__),
            '..',
            'nrc_spifpy',
            'config',
            '2DS.ini',
        ))
        with tempfile.TemporaryDirectory() as temp_dir:
            output = SPIFFile(os.path.join(temp_dir, 'spec.nc'), config)
            output.create_file()
            try:
                output.create_inst_group('2DS-H')
                core = output.instgrps['2DS-H']['core']
                output.create_variable(
                    core, 'tas', 'f4', ('Images',), data=[100.0, 100.0]
                )
                output.create_variable(
                    core,
                    'clock_counts',
                    'u8',
                    ('Images',),
                    data=[(1 << 32) - 5_000_000, 0],
                )
                core['buffer_index'][:] = [0, 1]

                self.spec.calc_image_times(['2DS-H'], output)

                numpy.testing.assert_array_equal(
                    core['image_sec'][:], [1, 1]
                )
                numpy.testing.assert_array_equal(
                    core['image_ns'][:], [0, 500_000_000]
                )
            finally:
                output.close()


if __name__ == '__main__':
    unittest.main()
