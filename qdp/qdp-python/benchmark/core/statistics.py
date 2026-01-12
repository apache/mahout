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

"""Statistical measurement framework for QDP benchmarks.

This module provides tools for collecting statistically rigorous benchmark
measurements with proper warmup, multiple runs, and comprehensive statistics.

Example:
    >>> from benchmark.core.statistics import StatisticalRunner, BenchmarkStats
    >>>
    >>> def my_benchmark():
    ...     # Some operation to benchmark
    ...     return result
    >>>
    >>> runner = StatisticalRunner(warmup_runs=3, measurement_runs=10)
    >>> stats = runner.run(my_benchmark)
    >>> print(f"Mean: {stats.mean:.3f} +/- {stats.std:.3f} ms")
    >>> print(f"P95: {stats.p95:.3f} ms")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np

T = TypeVar("T")


class OutlierMethod(Enum):
    """Methods for outlier detection and filtering."""

    NONE = "none"
    IQR = "iqr"  # Interquartile range method
    ZSCORE = "zscore"  # Z-score method (>3 std from mean)
    MAD = "mad"  # Median absolute deviation


@dataclass
class BenchmarkStats:
    """Comprehensive statistics for benchmark measurements.

    All timing values are in the same unit as provided to from_measurements()
    (typically milliseconds for latency benchmarks).

    Attributes:
        mean: Arithmetic mean of measurements.
        std: Standard deviation of measurements.
        min: Minimum value.
        max: Maximum value.
        median: Median (50th percentile).
        p5: 5th percentile.
        p25: 25th percentile (first quartile).
        p75: 75th percentile (third quartile).
        p95: 95th percentile.
        p99: 99th percentile.
        n_samples: Number of measurements used.
        n_outliers: Number of outliers filtered (if any).
        raw_measurements: Original measurements before filtering.
        unit: Unit of measurement (e.g., "ms", "s", "vectors/sec").
    """

    mean: float
    std: float
    min: float
    max: float
    median: float
    p5: float
    p25: float
    p75: float
    p95: float
    p99: float
    n_samples: int
    n_outliers: int = 0
    raw_measurements: List[float] = field(default_factory=list)
    unit: str = "ms"

    @classmethod
    def from_measurements(
        cls,
        measurements: List[float],
        outlier_method: OutlierMethod = OutlierMethod.NONE,
        outlier_threshold: float = 1.5,
        unit: str = "ms",
    ) -> "BenchmarkStats":
        """Create BenchmarkStats from a list of measurements.

        Args:
            measurements: List of timing measurements.
            outlier_method: Method for outlier detection.
            outlier_threshold: Threshold for outlier detection.
                - For IQR: multiplier for IQR (default 1.5 = standard)
                - For ZSCORE: number of standard deviations (default 3.0)
                - For MAD: multiplier for MAD (default 3.0)
            unit: Unit of measurement for display purposes.

        Returns:
            BenchmarkStats instance with computed statistics.

        Raises:
            ValueError: If measurements list is empty.
        """
        if not measurements:
            raise ValueError("Cannot compute statistics from empty measurements")

        raw = list(measurements)
        filtered = cls._filter_outliers(raw, outlier_method, outlier_threshold)
        n_outliers = len(raw) - len(filtered)

        arr = np.array(filtered)

        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=float(np.median(arr)),
            p5=float(np.percentile(arr, 5)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            n_samples=len(filtered),
            n_outliers=n_outliers,
            raw_measurements=raw,
            unit=unit,
        )

    @staticmethod
    def _filter_outliers(
        measurements: List[float],
        method: OutlierMethod,
        threshold: float,
    ) -> List[float]:
        """Filter outliers from measurements using specified method.

        Args:
            measurements: List of measurements to filter.
            method: Outlier detection method.
            threshold: Threshold for outlier detection.

        Returns:
            List of measurements with outliers removed.
        """
        if method == OutlierMethod.NONE or len(measurements) < 4:
            return measurements

        arr = np.array(measurements)

        if method == OutlierMethod.IQR:
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            mask = (arr >= lower_bound) & (arr <= upper_bound)

        elif method == OutlierMethod.ZSCORE:
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            if std == 0:
                return measurements
            z_scores = np.abs((arr - mean) / std)
            mask = z_scores <= threshold

        elif method == OutlierMethod.MAD:
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            if mad == 0:
                return measurements
            modified_z = 0.6745 * (arr - median) / mad
            mask = np.abs(modified_z) <= threshold

        else:
            return measurements

        filtered = arr[mask].tolist()
        # Ensure we keep at least half the measurements
        if len(filtered) < len(measurements) // 2:
            return measurements
        return filtered

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"BenchmarkStats(mean={self.mean:.3f} +/- {self.std:.3f} {self.unit}, "
            f"median={self.median:.3f}, p95={self.p95:.3f}, "
            f"n={self.n_samples})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "p5": self.p5,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "p99": self.p99,
            "n_samples": self.n_samples,
            "n_outliers": self.n_outliers,
            "unit": self.unit,
        }

    def summary_table(self, name: str = "Benchmark") -> str:
        """Generate a formatted summary table.

        Args:
            name: Name to display in the table header.

        Returns:
            Formatted string table with statistics.
        """
        lines = [
            f"{'=' * 50}",
            f"{name}",
            f"{'=' * 50}",
            f"  Mean:     {self.mean:12.3f} {self.unit}",
            f"  Std:      {self.std:12.3f} {self.unit}",
            f"  Min:      {self.min:12.3f} {self.unit}",
            f"  Max:      {self.max:12.3f} {self.unit}",
            f"  Median:   {self.median:12.3f} {self.unit}",
            f"{'-' * 50}",
            f"  P5:       {self.p5:12.3f} {self.unit}",
            f"  P25:      {self.p25:12.3f} {self.unit}",
            f"  P75:      {self.p75:12.3f} {self.unit}",
            f"  P95:      {self.p95:12.3f} {self.unit}",
            f"  P99:      {self.p99:12.3f} {self.unit}",
            f"{'-' * 50}",
            f"  Samples:  {self.n_samples:12d}",
        ]
        if self.n_outliers > 0:
            lines.append(f"  Outliers: {self.n_outliers:12d}")
        lines.append(f"{'=' * 50}")
        return "\n".join(lines)


class StatisticalRunner:
    """Runner for collecting statistically rigorous benchmark measurements.

    This class handles the mechanics of running benchmarks with proper warmup,
    collecting multiple measurements, and computing comprehensive statistics.

    Example:
        >>> runner = StatisticalRunner(warmup_runs=3, measurement_runs=10)
        >>> stats = runner.run(my_benchmark_fn, arg1, arg2, kwarg1=value)
        >>> print(stats.summary_table("My Benchmark"))

    Attributes:
        warmup_runs: Number of warmup iterations before measurement.
        measurement_runs: Number of measurement iterations.
        outlier_method: Method for outlier detection.
        outlier_threshold: Threshold for outlier detection.
        unit: Unit of measurement (default "ms").
    """

    def __init__(
        self,
        warmup_runs: int = 3,
        measurement_runs: int = 10,
        outlier_method: OutlierMethod = OutlierMethod.NONE,
        outlier_threshold: float = 1.5,
        unit: str = "ms",
    ):
        """Initialize the statistical runner.

        Args:
            warmup_runs: Number of warmup iterations (not measured).
            measurement_runs: Number of iterations to measure.
            outlier_method: Method for outlier detection and filtering.
            outlier_threshold: Threshold for outlier detection.
            unit: Unit of measurement for display purposes.

        Raises:
            ValueError: If warmup_runs < 0 or measurement_runs < 1.
        """
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")
        if measurement_runs < 1:
            raise ValueError("measurement_runs must be >= 1")

        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.unit = unit

    def run(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> BenchmarkStats:
        """Run the benchmark function and collect statistics.

        The function is first run warmup_runs times without measurement,
        then measurement_runs times with timing. Results are aggregated
        into BenchmarkStats.

        Args:
            fn: The function to benchmark.
            *args: Positional arguments to pass to fn.
            **kwargs: Keyword arguments to pass to fn.

        Returns:
            BenchmarkStats with comprehensive statistics.
        """
        # Warmup phase
        for _ in range(self.warmup_runs):
            fn(*args, **kwargs)

        # Measurement phase
        measurements: List[float] = []
        for _ in range(self.measurement_runs):
            start = time.perf_counter()
            fn(*args, **kwargs)
            end = time.perf_counter()

            # Convert to milliseconds if unit is "ms"
            elapsed = (end - start) * 1000 if self.unit == "ms" else (end - start)
            measurements.append(elapsed)

        return BenchmarkStats.from_measurements(
            measurements,
            outlier_method=self.outlier_method,
            outlier_threshold=self.outlier_threshold,
            unit=self.unit,
        )

    def run_with_setup(
        self,
        fn: Callable[..., T],
        setup: Callable[[], tuple],
        teardown: Optional[Callable[[], None]] = None,
    ) -> BenchmarkStats:
        """Run benchmark with per-iteration setup and teardown.

        This is useful when each benchmark iteration needs fresh state,
        such as clearing GPU caches between runs.

        Args:
            fn: The function to benchmark.
            setup: Function that returns (args, kwargs) for fn.
            teardown: Optional cleanup function called after each iteration.

        Returns:
            BenchmarkStats with comprehensive statistics.
        """
        # Warmup phase
        for _ in range(self.warmup_runs):
            args, kwargs = setup()
            fn(*args, **kwargs)
            if teardown:
                teardown()

        # Measurement phase
        measurements: List[float] = []
        for _ in range(self.measurement_runs):
            args, kwargs = setup()

            start = time.perf_counter()
            fn(*args, **kwargs)
            end = time.perf_counter()

            if teardown:
                teardown()

            elapsed = (end - start) * 1000 if self.unit == "ms" else (end - start)
            measurements.append(elapsed)

        return BenchmarkStats.from_measurements(
            measurements,
            outlier_method=self.outlier_method,
            outlier_threshold=self.outlier_threshold,
            unit=self.unit,
        )

    def run_timed(
        self,
        fn: Callable[..., float],
        *args: Any,
        **kwargs: Any,
    ) -> BenchmarkStats:
        """Run a function that returns its own timing measurement.

        This is useful when the function uses CUDA events or other
        internal timing mechanisms that are more accurate than wall-clock.

        Args:
            fn: Function that returns elapsed time in the configured unit.
            *args: Positional arguments to pass to fn.
            **kwargs: Keyword arguments to pass to fn.

        Returns:
            BenchmarkStats with comprehensive statistics.
        """
        # Warmup phase
        for _ in range(self.warmup_runs):
            fn(*args, **kwargs)

        # Measurement phase
        measurements: List[float] = []
        for _ in range(self.measurement_runs):
            elapsed = fn(*args, **kwargs)
            measurements.append(elapsed)

        return BenchmarkStats.from_measurements(
            measurements,
            outlier_method=self.outlier_method,
            outlier_threshold=self.outlier_threshold,
            unit=self.unit,
        )


def compare_stats(
    baseline: BenchmarkStats,
    comparison: BenchmarkStats,
    baseline_name: str = "Baseline",
    comparison_name: str = "Comparison",
) -> str:
    """Generate a comparison report between two benchmark results.

    Args:
        baseline: The baseline benchmark stats.
        comparison: The comparison benchmark stats.
        baseline_name: Display name for baseline.
        comparison_name: Display name for comparison.

    Returns:
        Formatted string with comparison statistics.
    """
    speedup = baseline.mean / comparison.mean if comparison.mean > 0 else float("inf")

    lines = [
        f"{'=' * 60}",
        f"Comparison: {comparison_name} vs {baseline_name}",
        f"{'=' * 60}",
        f"  {baseline_name:20s}: {baseline.mean:10.3f} +/- {baseline.std:.3f} {baseline.unit}",
        f"  {comparison_name:20s}: {comparison.mean:10.3f} +/- {comparison.std:.3f} {comparison.unit}",
        f"{'-' * 60}",
        f"  Speedup (mean):       {speedup:10.2f}x",
        f"  Speedup (median):     {baseline.median / comparison.median if comparison.median > 0 else float('inf'):10.2f}x",
        f"  Speedup (p95):        {baseline.p95 / comparison.p95 if comparison.p95 > 0 else float('inf'):10.2f}x",
        f"{'=' * 60}",
    ]
    return "\n".join(lines)
