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

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_QDP_PYTHON = Path(__file__).resolve().parents[2] / "qdp" / "qdp-python"
if str(_QDP_PYTHON) not in sys.path:
    sys.path.insert(0, str(_QDP_PYTHON))


def _install_local_benchmark_package() -> None:
    utils_path = _QDP_PYTHON / "benchmark" / "utils.py"
    utils_spec = importlib.util.spec_from_file_location("benchmark.utils", utils_path)
    assert utils_spec is not None
    assert utils_spec.loader is not None
    utils_module = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils_module)

    benchmark_package = types.ModuleType("benchmark")
    benchmark_package.__path__ = [str(_QDP_PYTHON / "benchmark")]
    setattr(benchmark_package, "utils", utils_module)

    sys.modules["benchmark"] = benchmark_package
    sys.modules["benchmark.utils"] = utils_module


def _load_module(name: str, relative_path: str):
    _install_local_benchmark_package()
    module_path = _QDP_PYTHON / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark_latency = _load_module(
    "qdp_benchmark_latency", "benchmark/benchmark_latency.py"
)
benchmark_throughput = _load_module(
    "qdp_benchmark_throughput", "benchmark/benchmark_throughput.py"
)


@pytest.mark.parametrize(
    ("module", "frameworks", "encoding_method"),
    [
        (benchmark_latency, ["mahout", "pennylane"], "angle"),
        (benchmark_latency, ["mahout", "qiskit-init"], "basis"),
        (benchmark_latency, ["mahout", "qiskit-statevector"], "iqp"),
        (benchmark_throughput, ["mahout", "qiskit"], "angle"),
        (benchmark_throughput, ["mahout", "mahout-amd"], "basis"),
        (benchmark_throughput, ["mahout", "pytorch-ref"], "iqp-z"),
    ],
)
def test_non_amplitude_cross_framework_combinations_are_rejected(
    module, frameworks, encoding_method
):
    with pytest.raises(ValueError, match="currently support non-amplitude encodings"):
        module.validate_framework_selection(frameworks, encoding_method)


@pytest.mark.parametrize("module", [benchmark_latency, benchmark_throughput])
@pytest.mark.parametrize("encoding_method", ["angle", "basis", "iqp", "iqp-z"])
def test_mahout_only_non_amplitude_runs_remain_allowed(module, encoding_method):
    assert module.validate_framework_selection(["mahout"], encoding_method) == [
        "mahout"
    ]


@pytest.mark.parametrize("module", [benchmark_latency, benchmark_throughput])
def test_amplitude_cross_framework_comparisons_remain_allowed(module):
    frameworks = ["mahout", "pennylane"]

    assert module.validate_framework_selection(frameworks, "amplitude") == frameworks


@pytest.mark.parametrize(
    ("module", "framework"),
    [
        (benchmark_latency, "pennylane"),
        (benchmark_latency, "qiskit-init"),
        (benchmark_throughput, "qiskit"),
        (benchmark_throughput, "mahout-amd"),
        (benchmark_throughput, "pytorch-ref"),
    ],
)
def test_non_amplitude_single_framework_runs_are_mahout_only(module, framework):
    with pytest.raises(ValueError, match="currently support non-amplitude encodings"):
        module.validate_framework_selection([framework], "basis")


def test_latency_main_rejects_invalid_non_amplitude_cli_combo(monkeypatch, capsys):
    monkeypatch.setattr(benchmark_latency.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_latency.py",
            "--frameworks",
            "mahout,pennylane",
            "--encoding-method",
            "angle",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        benchmark_latency.main()

    assert exc_info.value.code == 2
    assert "currently support non-amplitude encodings" in capsys.readouterr().err


def test_throughput_main_rejects_invalid_non_amplitude_cli_combo(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_throughput.py",
            "--frameworks",
            "mahout,qiskit",
            "--encoding-method",
            "basis",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        benchmark_throughput.main()

    assert exc_info.value.code == 2
    assert "currently support non-amplitude encodings" in capsys.readouterr().err


def test_latency_main_rejects_default_all_frameworks_for_non_amplitude(
    monkeypatch, capsys
):
    monkeypatch.setattr(benchmark_latency.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_latency.py", "--encoding-method", "iqp-z"],
    )

    with pytest.raises(SystemExit) as exc_info:
        benchmark_latency.main()

    assert exc_info.value.code == 2
    assert "currently support non-amplitude encodings" in capsys.readouterr().err


def test_throughput_main_rejects_default_all_frameworks_for_non_amplitude(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_throughput.py", "--encoding-method", "basis"],
    )

    with pytest.raises(SystemExit) as exc_info:
        benchmark_throughput.main()

    assert exc_info.value.code == 2
    assert "currently support non-amplitude encodings" in capsys.readouterr().err
