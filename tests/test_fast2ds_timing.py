import unittest
import numpy
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from nrc_spifpy.input.fast_2ds_file import Fast2DSFile

class TestFast2DSTiming(unittest.TestCase):
    def setUp(self):
        self.f2ds = Fast2DSFile('dummy.F2DS', 'F2DS', 10)
        self.f2ds.start_date = pd.Timestamp('2024-01-01 12:00:00')
        self.f2ds.datetimes = numpy.array([
            pd.Timestamp('2024-01-01 12:00:00'),
            pd.Timestamp('2024-01-01 12:00:01'),
            pd.Timestamp('2024-01-01 12:00:02'),
        ], dtype='datetime64[ns]')
        self.f2ds.tas = numpy.full(3, 100.0)
        self.f2ds.hk_datetimes = self.f2ds.datetimes.copy()
        self.f2ds.hk_tas = numpy.full(3, 100.0)
        self.f2ds.hk_counts = numpy.array(
            [1_000, 10_001_000, 20_001_000], dtype=numpy.uint64
        )

    def test_hk_anchored_probe_counter_calculation(self):
        timings = numpy.array([2_501_000, 15_001_000], dtype=numpy.uint64)
        buf_indices = numpy.array([0, 1], dtype=numpy.int64)

        sec, ns, quality = self.f2ds._calculate_particle_times(
            timings, buf_indices
        )

        self.assertEqual(sec[0], 0)
        self.assertAlmostEqual(ns[0], 250_000_000, delta=1)
        self.assertEqual(sec[1], 1)
        self.assertAlmostEqual(ns[1], 500_000_000, delta=1)
        numpy.testing.assert_array_equal(quality, [1, 1])

    def test_counter_elapsed_uses_mean_endpoint_tas(self):
        elapsed = self.f2ds._counter_elapsed_seconds(
            numpy.array([10_000_000], dtype=numpy.int64),
            numpy.array([100.0, 200.0]),
        )

        # 10,000,000 counts * 10 um/count = 100 m; 100 m / 150 m/s.
        numpy.testing.assert_allclose(elapsed, [0.0, 2.0 / 3.0])

    def test_48_bit_rollover(self):
        modulus = self.f2ds.TIMING_MODULUS
        self.f2ds.hk_counts = numpy.array(
            [modulus - 5_000_000, 5_000_000], dtype=numpy.uint64
        )
        self.f2ds.hk_datetimes = self.f2ds.datetimes[:2]
        self.f2ds.hk_tas = numpy.full(2, 100.0)

        sec, ns, quality = self.f2ds._calculate_particle_times(
            numpy.array([0], dtype=numpy.uint64),
            numpy.array([0], dtype=numpy.int64),
        )

        self.assertEqual(sec[0], 0)
        self.assertAlmostEqual(ns[0], 500_000_000, delta=1)
        self.assertEqual(quality[0], 1)

    def test_hk_timestamp_jitter_does_not_change_interarrival_time(self):
        self.f2ds.hk_datetimes = numpy.array([
            '2024-01-01T12:00:00.010',
            '2024-01-01T12:00:00.990',
            '2024-01-01T12:00:02.020',
        ], dtype='datetime64[ns]')

        sec, ns, quality = self.f2ds._calculate_particle_times(
            numpy.array([2_501_000, 15_001_000], dtype=numpy.uint64),
            numpy.array([0, 1], dtype=numpy.int64),
        )

        particle_seconds = sec.astype(numpy.float64) + ns / 1.0e9
        self.assertAlmostEqual(particle_seconds[0], 0.26, places=9)
        self.assertAlmostEqual(particle_seconds[1], 1.51, places=9)
        self.assertAlmostEqual(numpy.diff(particle_seconds)[0], 1.25, places=9)
        numpy.testing.assert_array_equal(quality, [1, 1])

    def test_missing_hk_uses_buffer_anchored_probe_counter(self):
        self.f2ds.hk_counts = numpy.array([], dtype=numpy.uint64)
        self.f2ds.hk_datetimes = numpy.array([], dtype='datetime64[ns]')
        self.f2ds.hk_tas = numpy.array([], dtype=numpy.float64)

        sec, ns, quality = self.f2ds._calculate_particle_times(
            numpy.array(
                [1_000_000, 3_500_000, 11_000_000], dtype=numpy.uint64
            ),
            numpy.array([0, 0, 1], dtype=numpy.int64),
        )

        particle_seconds = sec.astype(numpy.float64) + ns / 1.0e9
        numpy.testing.assert_allclose(particle_seconds, [0.0, 0.25, 1.0])
        numpy.testing.assert_array_equal(quality, [2, 2, 2])

    def test_single_particle_without_hk_uses_raw_buffer_time(self):
        self.f2ds.hk_counts = numpy.array([], dtype=numpy.uint64)
        self.f2ds.hk_datetimes = numpy.array([], dtype='datetime64[ns]')
        self.f2ds.hk_tas = numpy.array([], dtype=numpy.float64)

        sec, ns, quality = self.f2ds._calculate_particle_times(
            numpy.array([123], dtype=numpy.uint64),
            numpy.array([1], dtype=numpy.int64),
        )

        self.assertEqual(sec[0], 1)
        self.assertEqual(ns[0], 0)
        self.assertEqual(quality[0], 3)

    def test_buffer_counter_fallback_can_estimate_rate_without_tas(self):
        self.f2ds.hk_counts = numpy.array([], dtype=numpy.uint64)
        self.f2ds.hk_datetimes = numpy.array([], dtype='datetime64[ns]')
        self.f2ds.hk_tas = numpy.array([], dtype=numpy.float64)
        self.f2ds.tas[:] = numpy.nan

        sec, ns, quality = self.f2ds._calculate_particle_times(
            numpy.array([1, 3, 11, 13], dtype=numpy.uint64),
            numpy.array([0, 0, 1, 1], dtype=numpy.int64),
        )

        particle_seconds = sec.astype(numpy.float64) + ns / 1.0e9
        numpy.testing.assert_allclose(
            particle_seconds, [-0.1, 0.1, 0.9, 1.1]
        )
        numpy.testing.assert_array_equal(quality, [2, 2, 2, 2])

    def test_backwards_replacement_rechecks_adjacent_boundaries(self):
        particle_seconds = numpy.array([1.0, 3.0, 2.0, 4.0])
        timing_quality = numpy.ones(4, dtype=numpy.uint8)
        fallback_seconds = numpy.array([0.0, 0.5, 1.5, 4.0])
        fallback_quality = numpy.full(4, 2, dtype=numpy.uint8)

        self.f2ds._replace_backwards_times(
            particle_seconds,
            timing_quality,
            fallback_seconds,
            fallback_quality,
        )

        numpy.testing.assert_allclose(
            particle_seconds, [0.0, 0.5, 1.5, 4.0]
        )
        numpy.testing.assert_array_equal(timing_quality, [2, 2, 2, 1])
        self.assertTrue(numpy.all(numpy.diff(particle_seconds) >= 0.0))

if __name__ == "__main__":
    unittest.main()
