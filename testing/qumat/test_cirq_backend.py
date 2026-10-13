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
#

from collections.abc import Callable
from typing import Any

import cirq
import pytest

from qumat import cirq_backend
from qumat.cirq_backend import (
    apply_hadamard_gate,
    apply_rx_gate,
    create_empty_circuit,
    execute_circuit,
    get_final_state_vector,
    initialize_backend,
)

SEED = 17


def _hadamard_circuit() -> cirq.Circuit:
    circuit = create_empty_circuit(1)
    apply_hadamard_gate(circuit, 0)
    return circuit


def _record_calls(
    simulator: cirq.Simulator,
    method_name: str,
    calls: list[tuple[str, cirq.Simulator]],
) -> None:
    original: Callable[..., Any] = getattr(simulator, method_name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.append((method_name, simulator))
        return original(*args, **kwargs)

    setattr(simulator, method_name, wrapper)


def _initialized_simulator(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> cirq.Simulator:
    """Return initialize_backend()'s simulator, constructed with a fixed seed."""
    real_simulator = cirq.Simulator

    def factory(*args: Any, **kwargs: Any) -> cirq.Simulator:
        kwargs.setdefault("seed", seed)
        simulator = real_simulator(*args, **kwargs)
        setattr(simulator, "qumat_marker", "initialized")
        return simulator

    monkeypatch.setattr(cirq_backend.cirq, "Simulator", factory)
    backend = initialize_backend({"backend_options": {"simulator_type": "default"}})
    monkeypatch.setattr(cirq_backend.cirq, "Simulator", real_simulator)
    return backend


def test_execute_paths_reuse_initialized_simulator(monkeypatch):
    """Sampling and state-vector paths must call the simulator from initialize_backend."""
    backend = _initialized_simulator(monkeypatch, SEED)
    assert getattr(backend, "qumat_marker", None) == "initialized"

    calls: list[tuple[str, cirq.Simulator]] = []
    _record_calls(backend, "run", calls)
    _record_calls(backend, "run_sweep", calls)
    _record_calls(backend, "simulate", calls)

    measured = execute_circuit(
        _hadamard_circuit(),
        backend,
        {"backend_options": {"shots": 8}},
    )
    assert measured
    assert ("run", backend) in calls
    assert all(simulator is backend for _, simulator in calls)

    calls.clear()
    state = get_final_state_vector(
        _hadamard_circuit(),
        backend,
        {"backend_options": {}},
    )
    assert state is not None
    assert ("simulate", backend) in calls
    assert all(simulator is backend for _, simulator in calls)

    calls.clear()
    parameterized = create_empty_circuit(1)
    apply_rx_gate(parameterized, 0, "theta")
    parameter_config = {
        "backend_options": {"shots": 4},
        "parameter_values": {"theta": 0.3},
    }
    execute_circuit(parameterized, backend, parameter_config)
    assert ("run_sweep", backend) in calls
    assert all(simulator is backend for _, simulator in calls)

    calls.clear()
    swept_state = get_final_state_vector(parameterized, backend, parameter_config)
    assert swept_state is not None
    assert ("simulate", backend) in calls
    assert all(simulator is backend for _, simulator in calls)


def test_seeded_simulator_from_init_determines_samples(monkeypatch):
    """Measurement counts must follow the seed on the initialized simulator."""
    shots = 200
    circuit = _hadamard_circuit()
    config = {"backend_options": {"shots": shots}}

    first = execute_circuit(circuit, _initialized_simulator(monkeypatch, SEED), config)
    second = execute_circuit(circuit, _initialized_simulator(monkeypatch, SEED), config)
    other = execute_circuit(circuit, _initialized_simulator(monkeypatch, 99), config)

    assert first == second
    assert first != other
