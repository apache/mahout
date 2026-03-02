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
from braket.circuits import FreeParameter

from qumat.amazon_braket_backend import (
    initialize_backend,
    create_empty_circuit,
    apply_not_gate,
    apply_hadamard_gate,
    apply_cnot_gate,
    apply_toffoli_gate,
    apply_swap_gate,
    apply_cswap_gate,
    apply_pauli_x_gate,
    apply_pauli_y_gate,
    apply_pauli_z_gate,
    apply_t_gate,
    apply_rx_gate,
    apply_ry_gate,
    apply_rz_gate,
    apply_u_gate,
    execute_circuit,
    get_final_state_vector,
    calculate_prob_zero,
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
    circuit.parameters = []
    circuit.state_vector = MagicMock()

    mock_backend = MagicMock()
    mock_task = MagicMock()
    mock_result = MagicMock()

    mock_backend.run.return_value = mock_task
    mock_task.result.return_value = mock_result
    mock_result.values = [[1, 0]]

    config = {"backend_options": {}}

    result = get_final_state_vector(circuit, mock_backend, config)

    circuit.state_vector.assert_called_once()
    assert result == [1, 0]


def test_get_final_state_vector_with_parameters_full_branch():
    circuit = MagicMock()
    circuit.state_vector = MagicMock()

    param = MagicMock()
    param.name = "theta"
    circuit.parameters = [param]

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

    circuit.state_vector.assert_called_once()

    args, kwargs = mock_backend.run.call_args
    assert kwargs["shots"] == 0
    assert kwargs["inputs"] == {"theta": 0.5}

    assert result == [1, 0]


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
