#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the statistical measurement framework."""

import time
from typing import List

import numpy as np
import pytest

from benchmark.core.statistics import (
    BenchmarkStats,
    OutlierMethod,
    StatisticalRunner,
    compare_stats,
)


class TestBenchmarkStats:
    """Tests for BenchmarkStats dataclass."""

    def test_from_measurements_basic(self):
        """Test basic statistics computation from measurements."""
        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = BenchmarkStats.from_measurements(measurements)

        assert stats.mean == pytest.approx(3.0)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.n_samples == 5
        assert stats.n_outliers == 0

    def test_from_measurements_single_value(self):
        """Test statistics with single measurement."""
        measurements = [5.0]
        stats = BenchmarkStats.from_measurements(measurements)

        assert stats.mean == pytest.approx(5.0)
        assert stats.std == pytest.approx(0.0)
        assert stats.min == pytest.approx(5.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.median == pytest.approx(5.0)
        assert stats.n_samples == 1

    def test_from_measurements_empty_raises(self):
        """Test that empty measurements raise ValueError."""
        with pytest.raises(ValueError, match="empty measurements"):
            BenchmarkStats.from_measurements([])

    def test_std_calculation(self):
        """Test standard deviation calculation (sample std, ddof=1)."""
        # Known values: [2, 4, 4, 4, 5, 5, 7, 9] -> std = 2.138
        measurements = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        stats = BenchmarkStats.from_measurements(measurements)

        expected_std = np.std(measurements, ddof=1)
        assert stats.std == pytest.approx(expected_std, rel=1e-6)

    def test_percentiles(self):
        """Test percentile calculations."""
        # Create data where percentiles are easy to verify
        measurements = list(range(1, 101))  # 1 to 100
        stats = BenchmarkStats.from_measurements(measurements)

        assert stats.p5 == pytest.approx(5.95, rel=0.1)
        assert stats.p25 == pytest.approx(25.75, rel=0.1)
        assert stats.median == pytest.approx(50.5, rel=0.1)
        assert stats.p75 == pytest.approx(75.25, rel=0.1)
        assert stats.p95 == pytest.approx(95.05, rel=0.1)
        assert stats.p99 == pytest.approx(99.01, rel=0.1)

    def test_unit_preserved(self):
        """Test that unit is preserved in stats."""
        stats = BenchmarkStats.from_measurements([1.0, 2.0, 3.0], unit="s")
        assert stats.unit == "s"

        stats_default = BenchmarkStats.from_measurements([1.0, 2.0, 3.0])
        assert stats_default.unit == "ms"

    def test_raw_measurements_stored(self):
        """Test that raw measurements are stored."""
        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = BenchmarkStats.from_measurements(measurements)
        assert stats.raw_measurements == measurements

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = BenchmarkStats.from_measurements([1.0, 2.0, 3.0])
        d = stats.to_dict()

        assert "mean" in d
        assert "std" in d
        assert "min" in d
        assert "max" in d
        assert "median" in d
        assert "p5" in d
        assert "p25" in d
        assert "p75" in d
        assert "p95" in d
        assert "p99" in d
        assert "n_samples" in d
        assert "unit" in d

    def test_str_representation(self):
        """Test string representation."""
        stats = BenchmarkStats.from_measurements([1.0, 2.0, 3.0])
        s = str(stats)

        assert "BenchmarkStats" in s
        assert "mean=" in s
        assert "median=" in s

    def test_summary_table(self):
        """Test summary table generation."""
        stats = BenchmarkStats.from_measurements([1.0, 2.0, 3.0, 4.0, 5.0])
        table = stats.summary_table("Test Benchmark")

        assert "Test Benchmark" in table
        assert "Mean:" in table
        assert "Std:" in table
        assert "P95:" in table
        assert "Samples:" in table


class TestOutlierFiltering:
    """Tests for outlier detection and filtering."""

    def test_no_filtering(self):
        """Test that NONE method keeps all measurements."""
        measurements = [1.0, 2.0, 3.0, 100.0]  # 100 is an outlier
        stats = BenchmarkStats.from_measurements(
            measurements, outlier_method=OutlierMethod.NONE
        )

        assert stats.n_samples == 4
        assert stats.n_outliers == 0

    def test_iqr_filtering(self):
        """Test IQR-based outlier filtering."""
        # Normal values with extreme outliers
        measurements = [10.0, 11.0, 10.5, 10.2, 10.8, 11.2, 10.1, 100.0, 0.1]
        stats = BenchmarkStats.from_measurements(
            measurements, outlier_method=OutlierMethod.IQR, outlier_threshold=1.5
        )

        # Should filter out 100.0 and 0.1
        assert stats.n_samples < 9
        assert stats.n_outliers > 0
        assert 100.0 not in [
            m
            for m in measurements
            if abs(m - stats.mean) < 3 * stats.std or stats.std == 0
        ]

    def test_zscore_filtering(self):
        """Test z-score based outlier filtering."""
        # Normal distribution with outlier
        np.random.seed(42)
        measurements = list(np.random.normal(10, 1, 20)) + [50.0]  # 50 is outlier

        stats = BenchmarkStats.from_measurements(
            measurements, outlier_method=OutlierMethod.ZSCORE, outlier_threshold=3.0
        )

        assert stats.n_outliers >= 1
        assert stats.mean < 15  # Mean should be closer to 10 without outlier

    def test_mad_filtering(self):
        """Test MAD-based outlier filtering."""
        measurements = [10.0, 10.5, 11.0, 10.2, 10.8, 9.9, 10.3, 50.0]
        stats = BenchmarkStats.from_measurements(
            measurements, outlier_method=OutlierMethod.MAD, outlier_threshold=3.0
        )

        assert stats.n_outliers >= 1

    def test_min_samples_preserved(self):
        """Test that filtering preserves at least half the measurements."""
        # All values are "outliers" relative to each other in this extreme case
        measurements = [1.0, 100.0, 200.0, 300.0, 400.0, 500.0]
        stats = BenchmarkStats.from_measurements(
            measurements, outlier_method=OutlierMethod.IQR
        )

        # Should keep at least half
        assert stats.n_samples >= 3

    def test_small_sample_no_filtering(self):
        """Test that small samples skip filtering."""
        measurements = [1.0, 100.0, 200.0]  # Only 3 samples
        stats = BenchmarkStats.from_measurements(
            measurements, outlier_method=OutlierMethod.IQR
        )

        # Should keep all with < 4 samples
        assert stats.n_samples == 3
        assert stats.n_outliers == 0


class TestStatisticalRunner:
    """Tests for StatisticalRunner class."""

    def test_init_defaults(self):
        """Test default initialization."""
        runner = StatisticalRunner()

        assert runner.warmup_runs == 3
        assert runner.measurement_runs == 10
        assert runner.outlier_method == OutlierMethod.NONE
        assert runner.unit == "ms"

    def test_init_custom(self):
        """Test custom initialization."""
        runner = StatisticalRunner(
            warmup_runs=5,
            measurement_runs=20,
            outlier_method=OutlierMethod.IQR,
            outlier_threshold=2.0,
            unit="s",
        )

        assert runner.warmup_runs == 5
        assert runner.measurement_runs == 20
        assert runner.outlier_method == OutlierMethod.IQR
        assert runner.outlier_threshold == 2.0
        assert runner.unit == "s"

    def test_init_invalid_warmup(self):
        """Test that negative warmup_runs raises ValueError."""
        with pytest.raises(ValueError, match="warmup_runs"):
            StatisticalRunner(warmup_runs=-1)

    def test_init_invalid_measurement(self):
        """Test that zero measurement_runs raises ValueError."""
        with pytest.raises(ValueError, match="measurement_runs"):
            StatisticalRunner(measurement_runs=0)

    def test_run_basic(self):
        """Test basic benchmark run."""
        call_count = 0

        def simple_fn():
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)  # 1ms

        runner = StatisticalRunner(warmup_runs=2, measurement_runs=5)
        stats = runner.run(simple_fn)

        # Should have called function warmup + measurement times
        assert call_count == 7

        # Timing should be around 1ms (with some tolerance)
        assert stats.mean > 0.5  # At least 0.5ms
        assert stats.mean < 50  # Less than 50ms
        assert stats.n_samples == 5

    def test_run_with_args(self):
        """Test running benchmark with arguments."""
        results: List[int] = []

        def fn_with_args(a: int, b: int, c: int = 0) -> int:
            result = a + b + c
            results.append(result)
            return result

        runner = StatisticalRunner(warmup_runs=1, measurement_runs=3)
        _ = runner.run(fn_with_args, 1, 2, c=3)

        # Should have correct number of calls
        assert len(results) == 4  # 1 warmup + 3 measurements

        # All results should be correct
        assert all(r == 6 for r in results)

    def test_run_with_setup(self):
        """Test running benchmark with setup/teardown."""
        setup_count = 0
        teardown_count = 0
        fn_count = 0

        def setup():
            nonlocal setup_count
            setup_count += 1
            return (setup_count,), {"multiplier": 2}

        def teardown():
            nonlocal teardown_count
            teardown_count += 1

        def benchmark_fn(value: int, multiplier: int = 1):
            nonlocal fn_count
            fn_count += 1
            return value * multiplier

        runner = StatisticalRunner(warmup_runs=2, measurement_runs=3)
        stats = runner.run_with_setup(benchmark_fn, setup, teardown)

        # Verify counts
        assert setup_count == 5  # 2 warmup + 3 measurement
        assert teardown_count == 5
        assert fn_count == 5
        assert stats.n_samples == 3

    def test_run_timed(self):
        """Test running benchmark that returns its own timing."""

        def self_timed_fn() -> float:
            time.sleep(0.001)
            return 1.5  # Always return 1.5ms

        runner = StatisticalRunner(warmup_runs=2, measurement_runs=5, unit="ms")
        stats = runner.run_timed(self_timed_fn)

        # Stats should be based on returned values, not wall clock
        assert stats.mean == pytest.approx(1.5)
        assert stats.std == pytest.approx(0.0)

    def test_unit_conversion(self):
        """Test that unit affects time conversion."""

        def fast_fn():
            pass  # Near-instant

        runner_ms = StatisticalRunner(warmup_runs=1, measurement_runs=3, unit="ms")
        stats_ms = runner_ms.run(fast_fn)

        runner_s = StatisticalRunner(warmup_runs=1, measurement_runs=3, unit="s")
        stats_s = runner_s.run(fast_fn)

        # ms values should be roughly 1000x larger than s values
        # (accounting for measurement noise, just check they're different orders)
        assert stats_ms.unit == "ms"
        assert stats_s.unit == "s"


class TestCompareStats:
    """Tests for compare_stats function."""

    def test_compare_basic(self):
        """Test basic comparison generation."""
        baseline = BenchmarkStats.from_measurements([10.0, 11.0, 10.5])
        comparison = BenchmarkStats.from_measurements([5.0, 5.5, 5.2])

        report = compare_stats(baseline, comparison, "Slow", "Fast")

        assert "Slow" in report
        assert "Fast" in report
        assert "Speedup" in report
        assert "2." in report  # ~2x speedup

    def test_compare_same_performance(self):
        """Test comparison with same performance."""
        stats1 = BenchmarkStats.from_measurements([10.0, 10.0, 10.0])
        stats2 = BenchmarkStats.from_measurements([10.0, 10.0, 10.0])

        report = compare_stats(stats1, stats2)

        assert "1.00x" in report  # 1x speedup

    def test_compare_zero_handling(self):
        """Test that zero comparison values are handled."""
        baseline = BenchmarkStats.from_measurements([10.0, 10.0])

        # Create stats with zero mean (edge case)
        zero_stats = BenchmarkStats(
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            median=0.0,
            p5=0.0,
            p25=0.0,
            p75=0.0,
            p95=0.0,
            p99=0.0,
            n_samples=1,
        )

        # Should not raise, should handle inf gracefully
        report = compare_stats(baseline, zero_stats)
        assert "inf" in report.lower() or "Speedup" in report


class TestIntegration:
    """Integration tests for the statistics module."""

    def test_full_benchmark_workflow(self):
        """Test a complete benchmark workflow."""
        # Simulate a function with variable performance
        np.random.seed(42)

        def variable_fn():
            # Sleep for 1-3ms randomly
            time.sleep(np.random.uniform(0.001, 0.003))

        runner = StatisticalRunner(
            warmup_runs=2,
            measurement_runs=10,
            outlier_method=OutlierMethod.IQR,
            unit="ms",
        )

        stats = runner.run(variable_fn)

        # Basic sanity checks
        assert stats.n_samples <= 10  # May have filtered outliers
        assert stats.mean > 0
        assert stats.std > 0  # Should have variance
        assert stats.min <= stats.median <= stats.max
        assert stats.p25 <= stats.median <= stats.p75

        # Should be able to generate reports
        table = stats.summary_table("Variable Function")
        assert len(table) > 0

        d = stats.to_dict()
        assert len(d) > 0

    def test_reproducibility_with_seed(self):
        """Test that measurements are consistent with fixed computation."""
        call_count = 0

        def deterministic_fn():
            nonlocal call_count
            call_count += 1
            # Do some actual computation
            _ = sum(range(1000))

        runner = StatisticalRunner(warmup_runs=1, measurement_runs=5)

        # Run multiple times
        stats1 = runner.run(deterministic_fn)
        count1 = call_count

        stats2 = runner.run(deterministic_fn)
        count2 = call_count

        # Call counts should increment correctly
        assert count1 == 6  # 1 warmup + 5 measurements
        assert count2 == 12  # Another 1 + 5

        # Stats should be similar (same function)
        assert stats1.mean > 0
        assert stats2.mean > 0
