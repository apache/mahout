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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from braket.circuits import FreeParameter

from qumat.amazon_braket_backend import (
    apply_cnot_gate,
    apply_cswap_gate,
    apply_hadamard_gate,
    apply_not_gate,
    apply_pauli_x_gate,
    apply_pauli_y_gate,
    apply_pauli_z_gate,
    apply_rx_gate,
    apply_ry_gate,
    apply_rz_gate,
    apply_swap_gate,
    apply_t_gate,
    apply_toffoli_gate,
    apply_u_gate,
    calculate_prob_zero,
    create_empty_circuit,
    execute_circuit,
    get_final_state_vector,
    initialize_backend,
)

# initialize_backend


def test_initialize_backend_local():
    with patch("qumat.amazon_braket_backend.LocalSimulator") as mock_local:
        config = {"backend_options": {"simulator_type": "local"}}
        initialize_backend(config)
        mock_local.assert_called_once()


def test_initialize_backend_default():
    with patch("qumat.amazon_braket_backend.AwsDevice") as mock_device:
        config = {"backend_options": {"simulator_type": "default"}}
        initialize_backend(config)
        mock_device.assert_called_once()


def test_initialize_backend_invalid_type():
    with patch("qumat.amazon_braket_backend.AwsDevice") as mock_device:
        config = {"backend_options": {"simulator_type": "invalid"}}
        initialize_backend(config)
        mock_device.assert_called_once()


def test_initialize_backend_with_region():
    with (
        patch("qumat.amazon_braket_backend.boto3.Session") as mock_boto,
        patch("qumat.amazon_braket_backend.AwsSession") as mock_aws_session,
        patch("qumat.amazon_braket_backend.AwsDevice") as mock_device,
    ):
        config = {
            "backend_options": {
                "simulator_type": "default",
                "region": "us-west-2",
            }
        }

        initialize_backend(config)

        mock_boto.assert_called_once_with(region_name="us-west-2")

        # Ensure boto3 session passed into AwsSession
        mock_aws_session.assert_called_once_with(boto_session=mock_boto.return_value)

        # Ensure AwsSession passed into AwsDevice
        mock_device.assert_called_once()
        _, device_kwargs = mock_device.call_args
        assert device_kwargs.get("aws_session") is mock_aws_session.return_value


# create_empty_circuit


def test_create_empty_circuit_with_qubits():
    circuit = create_empty_circuit(3)
    assert len(circuit.instructions) == 3


def test_create_empty_circuit_without_qubits():
    circuit = create_empty_circuit()
    assert len(circuit.instructions) == 0


# Basic gate wrappers


def test_basic_gate_wrappers():
    circuit = MagicMock()

    apply_not_gate(circuit, 0)
    apply_hadamard_gate(circuit, 1)
    apply_cnot_gate(circuit, 0, 1)
    apply_toffoli_gate(circuit, 0, 1, 2)
    apply_swap_gate(circuit, 0, 1)
    apply_cswap_gate(circuit, 0, 1, 2)
    apply_pauli_x_gate(circuit, 0)
    apply_pauli_y_gate(circuit, 1)
    apply_pauli_z_gate(circuit, 2)
    apply_t_gate(circuit, 0)

    circuit.x.assert_called()
    circuit.h.assert_called()
    circuit.cnot.assert_called()
    circuit.ccnot.assert_called()
    circuit.swap.assert_called()
    circuit.cswap.assert_called()
    circuit.y.assert_called()
    circuit.z.assert_called()
    circuit.t.assert_called()


# RX / RY / RZ


def test_apply_rx_gate_numeric():
    circuit = MagicMock()
    apply_rx_gate(circuit, 0, 1.23)
    circuit.rx.assert_called_once_with(0, 1.23)


def test_apply_rx_gate_parameter():
    circuit = MagicMock()
    apply_rx_gate(circuit, 0, "alpha")
    circuit.rx.assert_called_once()
    args = circuit.rx.call_args[0]
    assert isinstance(args[1], FreeParameter)


def test_apply_ry_gate_numeric():
    circuit = MagicMock()
    apply_ry_gate(circuit, 0, 1.57)
    circuit.ry.assert_called_once_with(0, 1.57)


def test_apply_ry_gate_parameter():
    circuit = MagicMock()
    apply_ry_gate(circuit, 0, "theta")
    circuit.ry.assert_called_once()
    args = circuit.ry.call_args[0]
    assert isinstance(args[1], FreeParameter)


def test_apply_rz_gate_numeric():
    circuit = MagicMock()
    apply_rz_gate(circuit, 0, 3.14)
    circuit.rz.assert_called_once_with(0, 3.14)


def test_apply_rz_gate_parameter():
    circuit = MagicMock()
    apply_rz_gate(circuit, 0, "phi")
    circuit.rz.assert_called_once()
    args = circuit.rz.call_args[0]
    assert isinstance(args[1], FreeParameter)


def test_apply_u_gate_sequence():
    circuit = MagicMock()
    apply_u_gate(circuit, 0, 0.5, 0.3, 0.1)

    circuit.rz.assert_any_call(0, 0.1)
    circuit.ry.assert_called_once_with(0, 0.5)
    circuit.rz.assert_any_call(0, 0.3)


@pytest.mark.parametrize(
    ("apply_fn", "method"),
    [(apply_rx_gate, "rx"), (apply_ry_gate, "ry"), (apply_rz_gate, "rz")],
)
@pytest.mark.parametrize("angle", [np.float32(0.5), np.float64(0.5), np.int64(2), 2])
def test_rotation_gates_accept_numpy_scalars(apply_fn, method, angle):
    circuit = MagicMock()
    apply_fn(circuit, 0, angle)
    passed = getattr(circuit, method).call_args[0][1]
    assert type(passed) is float
    assert passed == pytest.approx(float(angle))


def test_apply_u_gate_accepts_numpy_scalars():
    circuit = MagicMock()
    apply_u_gate(circuit, 0, np.float32(0.5), np.int64(1), np.float64(0.1))
    angles = [c[0][1] for c in circuit.rz.call_args_list]
    angles += [c[0][1] for c in circuit.ry.call_args_list]
    assert all(type(a) is float for a in angles)


def test_numpy_scalar_parameter_values_execute_on_local_simulator():
    """np.float32 bound values used to fail with
    "Cannot wrap ... into literal type"."""
    from qumat import QuMat

    def run(value):
        qumat = QuMat(
            {
                "backend_name": "amazon_braket",
                "backend_options": {"simulator_type": "local", "shots": 10},
            }
        )
        qumat.create_empty_circuit(num_qubits=1)
        qumat.apply_rx_gate(0, "theta")
        counts = qumat.execute_circuit({"theta": value})
        state = qumat.get_final_state_vector()
        return counts, np.asarray(state)

    counts, state32 = run(np.float32(np.pi))
    assert dict(counts) == {"1": 10}
    _, state64 = run(np.pi)
    np.testing.assert_allclose(state32, state64, atol=1e-6)


# execute_circuit


def test_execute_circuit_without_parameters():
    circuit = MagicMock()
    circuit.parameters = []

    mock_backend = MagicMock()
    mock_task = MagicMock()
    mock_result = MagicMock()

    mock_backend.run.return_value = mock_task
    mock_task.result.return_value = mock_result
    mock_result.measurement_counts = {"00": 1}

    config = {"backend_options": {"shots": 1}}

    result = execute_circuit(circuit, mock_backend, config)

    assert result == {"00": 1}
    mock_backend.run.assert_called_once()


def test_execute_circuit_with_parameters():
    circuit = MagicMock()
    param = MagicMock()
    param.name = "theta"
    circuit.parameters = [param]

    mock_backend = MagicMock()
    mock_task = MagicMock()
    mock_result = MagicMock()

    mock_backend.run.return_value = mock_task
    mock_task.result.return_value = mock_result
    mock_result.measurement_counts = {"00": 1}

    config = {
        "backend_options": {"shots": 1},
        "parameter_values": {"theta": 0.5, "extra": 0.9},  # extra should be ignored
    }

    result = execute_circuit(circuit, mock_backend, config)

    mock_backend.run.assert_called_once()

    # Ensure only valid circuit parameters are passed
    _, kwargs = mock_backend.run.call_args
    assert kwargs["inputs"] == {"theta": 0.5}

    # Ensure backend result is returned
    assert result == {"00": 1}


# get_final_state_vector


def test_get_final_state_vector_without_parameters():
    circuit = MagicMock()
    working = circuit.copy.return_value
    working.parameters = []

    mock_backend = MagicMock()
    mock_task = MagicMock()
    mock_result = MagicMock()

    mock_backend.run.return_value = mock_task
    mock_task.result.return_value = mock_result
    mock_result.values = [[1, 0]]

    config = {"backend_options": {}}

    result = get_final_state_vector(circuit, mock_backend, config)

    working.state_vector.assert_called_once()
    circuit.state_vector.assert_not_called()
    mock_backend.run.assert_called_once_with(working, shots=0)
    assert result == [1, 0]


def test_get_final_state_vector_with_parameters_full_branch():
    circuit = MagicMock()
    working = circuit.copy.return_value

    param = MagicMock()
    param.name = "theta"
    working.parameters = [param]

    mock_backend = MagicMock()
    mock_task = MagicMock()
    mock_result = MagicMock()

    mock_backend.run.return_value = mock_task
    mock_task.result.return_value = mock_result
    mock_result.values = [[1, 0]]

    config = {
        "backend_options": {},
        "parameter_values": {"theta": 0.5, "unused": 99},
    }

    result = get_final_state_vector(circuit, mock_backend, config)

    working.state_vector.assert_called_once()
    circuit.state_vector.assert_not_called()

    args, kwargs = mock_backend.run.call_args
    assert args == (working,)
    assert kwargs["shots"] == 0
    assert kwargs["inputs"] == {"theta": 0.5}

    assert result == [1, 0]


def test_get_final_state_vector_does_not_block_later_shot_runs():
    """Regression: the state-vector query must not leave a StateVector
    result type on the user's circuit, or the next shot-based run fails."""
    from qumat import QuMat

    qumat = QuMat(
        {
            "backend_name": "amazon_braket",
            "backend_options": {"simulator_type": "local", "shots": 50},
        }
    )
    qumat.create_empty_circuit(num_qubits=2)
    qumat.apply_not_gate(0)

    state = qumat.get_final_state_vector()
    assert abs(state[2]) == pytest.approx(1.0)  # |10>, qubit 0 is leftmost
    assert qumat.circuit is not None
    assert qumat.circuit.result_types == []

    counts = qumat.execute_circuit()
    assert dict(counts) == {"10": 50}

    # And the other order, twice, still works.
    qumat.get_final_state_vector()
    assert qumat.circuit is not None
    assert qumat.circuit.result_types == []


# calculate_prob_zero


def test_calculate_prob_zero_basic():
    results = {"00": 3, "10": 1}
    prob = calculate_prob_zero(results, 0, 2)
    assert prob == 3 / 4


def test_calculate_prob_zero_list_input():
    results = [{"00": 2, "01": 2}]
    prob = calculate_prob_zero(results, 0, 2)
    assert prob == 1.0


def test_calculate_prob_zero_zero_shots():
    results = {}
    prob = calculate_prob_zero(results, 0, 1)
    assert prob == 0.0
