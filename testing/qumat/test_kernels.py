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
"""Tests for qumat.kernels (quantum fidelity kernels, issues #456 / #458)."""

import numpy as np
import pytest

from qumat import AngleFeatureMap, QuMat, ZZFeatureMap, fidelity_kernel, kernel_matrix
from qumat.kernels import GateOp, kernel_from_states

from ..utils import TESTING_BACKENDS
from ..utils import get_backend_config as _get_backend_config

# Shot-based methods. Backends are not seeded through QuMat, so the
# tolerances below are generous: with 4000 shots the standard error of
# P(0) is <= 0.008, i.e. <= 0.016 on the swap-test kernel (2 * P0 - 1).
SHOTS = 4000
SHOT_TOL = 0.1
SHOT_METHODS = ["compute_uncompute", "swap_test"]


def get_backend_config(backend_name: str) -> dict:
    config = _get_backend_config(backend_name)
    assert config is not None, f"unknown backend {backend_name}"
    return config


def analytic_angle_kernel(x, y, scale=1.0):
    """Closed form for AngleFeatureMap: prod_k cos^2(scale*(x_k - y_k)/2)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.prod(np.cos(scale * (x - y) / 2.0) ** 2))


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(1234)
    return rng.uniform(0.0, np.pi, size=(4, 2))


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestStatevectorKernel:
    """Exact (statevector) kernel on every backend."""

    def test_orthogonal_inputs_give_zero(self, backend_name):
        """RY(0)|0> = |0> and RY(pi)|0> = |1> are orthogonal.

        This is the prototype case from issue #456 (tdnaugle), where Qiskit
        returned a tiny non-zero value. Anything left is float round-off.
        """
        fm = AngleFeatureMap(num_qubits=2)
        k = fidelity_kernel(
            [0.0, 0.0], [np.pi, np.pi], fm, get_backend_config(backend_name)
        )
        assert k == pytest.approx(0.0, abs=1e-12)

    def test_identical_inputs_give_one(self, backend_name):
        fm = ZZFeatureMap(num_qubits=2, reps=1)
        x = [0.3, 1.7]
        k = fidelity_kernel(x, x, fm, get_backend_config(backend_name))
        assert k == pytest.approx(1.0, abs=1e-6)

    def test_matches_analytic_angle_kernel(self, backend_name, small_dataset):
        fm = AngleFeatureMap(num_qubits=2)
        gram = kernel_matrix(
            small_dataset,
            feature_map=fm,
            backend_config=get_backend_config(backend_name),
        )
        expected = np.array(
            [
                [analytic_angle_kernel(a, b) for b in small_dataset]
                for a in small_dataset
            ]
        )
        np.testing.assert_allclose(gram, expected, atol=1e-6)

    def test_gram_matrix_is_symmetric_psd_unit_diagonal(
        self, backend_name, small_dataset
    ):
        fm = ZZFeatureMap(num_qubits=2, reps=1)
        gram = kernel_matrix(
            small_dataset,
            feature_map=fm,
            backend_config=get_backend_config(backend_name),
        )
        assert gram.shape == (4, 4)
        np.testing.assert_array_equal(gram, gram.T)
        np.testing.assert_array_equal(np.diag(gram), np.ones(4))
        assert np.all((gram >= 0.0) & (gram <= 1.0))
        # A Gram matrix of inner products is PSD (up to float error).
        assert np.linalg.eigvalsh(gram).min() > -1e-6

    def test_rectangular_kernel_matrix(self, backend_name, small_dataset):
        """kernel_matrix(X_test, X_train) has shape (n_test, n_train)."""
        fm = AngleFeatureMap(num_qubits=2)
        train, test = small_dataset[:3], small_dataset[3:]
        gram = kernel_matrix(
            test, train, feature_map=fm, backend_config=get_backend_config(backend_name)
        )
        assert gram.shape == (1, 3)
        expected = [analytic_angle_kernel(test[0], b) for b in train]
        np.testing.assert_allclose(gram[0], expected, atol=1e-6)


class TestCrossBackendAgreement:
    """Kernels must not depend on a backend's qubit-ordering convention."""

    def test_raw_statevectors_differ_but_kernels_agree(self):
        """Qiskit is little-endian, Cirq big-endian: amplitudes are permuted.

        Both vectors describe the same physical state, so the overlap of two
        states from the *same* backend is ordering-invariant.
        """
        fm = ZZFeatureMap(num_qubits=2, reps=1)
        x = [0.4, 2.1]
        vectors = {}
        for name in ("qiskit", "cirq"):
            qumat = QuMat(get_backend_config(name))
            qumat.create_empty_circuit(num_qubits=2)
            AngleFeatureMap(2).apply(qumat, [0.0, np.pi])  # |q0=0, q1=1>
            vectors[name] = np.asarray(qumat.get_final_state_vector())
        # Qiskit puts q1 in the high bit (index 2), Cirq in the low bit (1).
        assert abs(vectors["qiskit"][2]) == pytest.approx(1.0, abs=1e-6)
        assert abs(vectors["cirq"][1]) == pytest.approx(1.0, abs=1e-6)

        y = [1.5, 0.2]
        values = [
            fidelity_kernel(x, y, fm, get_backend_config(b)) for b in TESTING_BACKENDS
        ]
        assert max(values) - min(values) < 1e-6

    def test_qiskit_and_cirq_gram_matrices_agree(self, small_dataset):
        fm = ZZFeatureMap(num_qubits=2, reps=2)
        grams = [
            kernel_matrix(
                small_dataset, feature_map=fm, backend_config=get_backend_config(b)
            )
            for b in ("qiskit", "cirq")
        ]
        np.testing.assert_allclose(grams[0], grams[1], atol=1e-6)

    def test_zz_map_matches_qiskit_library(self, small_dataset):
        """Our ZZFeatureMap equals qiskit's zz_feature_map up to global phase."""
        library = pytest.importorskip("qiskit.circuit.library")
        from qiskit.quantum_info import Statevector

        if not hasattr(library, "zz_feature_map"):
            pytest.skip("qiskit.circuit.library.zz_feature_map not available")
        reference = library.zz_feature_map(2, reps=2, entanglement="full")
        states = [
            Statevector(reference.assign_parameters(x)).data for x in small_dataset
        ]
        expected = kernel_from_states(states)
        gram = kernel_matrix(
            small_dataset,
            feature_map=ZZFeatureMap(2, reps=2, entanglement="full"),
            backend_config=get_backend_config("qiskit"),
        )
        np.testing.assert_allclose(gram, expected, atol=1e-9)


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
@pytest.mark.parametrize("method", SHOT_METHODS)
class TestShotBasedKernel:
    """compute-uncompute and swap-test estimators (statistical tolerances)."""

    def test_estimate_matches_analytic(self, backend_name, method):
        fm = AngleFeatureMap(num_qubits=2)
        x, y = [0.2, 1.1], [1.4, 0.5]
        k = fidelity_kernel(
            x,
            y,
            fm,
            get_backend_config(backend_name),
            method=method,
            shots=SHOTS,
        )
        assert k == pytest.approx(analytic_angle_kernel(x, y), abs=SHOT_TOL)

    def test_orthogonal_and_identical(self, backend_name, method):
        fm = AngleFeatureMap(num_qubits=1)
        config = get_backend_config(backend_name)
        same = fidelity_kernel([0.7], [0.7], fm, config, method=method, shots=SHOTS)
        orth = fidelity_kernel([0.0], [np.pi], fm, config, method=method, shots=SHOTS)
        assert same == pytest.approx(1.0, abs=SHOT_TOL)
        assert orth == pytest.approx(0.0, abs=SHOT_TOL)

    def test_symmetric_gram_matrix(self, backend_name, method):
        fm = ZZFeatureMap(num_qubits=2, reps=1)
        data = np.array([[0.1, 0.9], [1.2, 2.5], [2.8, 0.4]])
        gram = kernel_matrix(
            data,
            feature_map=fm,
            backend_config=get_backend_config(backend_name),
            method=method,
            shots=SHOTS,
        )
        exact = kernel_matrix(
            data, feature_map=fm, backend_config=get_backend_config(backend_name)
        )
        np.testing.assert_array_equal(gram, gram.T)
        np.testing.assert_array_equal(np.diag(gram), np.ones(3))
        np.testing.assert_allclose(gram, exact, atol=SHOT_TOL)


class TestFeatureMaps:
    def test_inverse_undoes_feature_map(self):
        """U(x)^dagger U(x) |0> = |0>, for the compute-uncompute method."""
        fm = ZZFeatureMap(num_qubits=3, reps=2, entanglement="circular")
        qumat = QuMat(get_backend_config("qiskit"))
        qumat.create_empty_circuit(num_qubits=3)
        x = [0.3, 1.2, 2.2]
        fm.apply(qumat, x)
        fm.apply(qumat, x, inverse=True)
        state = np.asarray(qumat.get_final_state_vector())
        assert abs(state[0]) ** 2 == pytest.approx(1.0, abs=1e-9)

    def test_offset_places_map_on_other_register(self):
        fm = AngleFeatureMap(num_qubits=1)
        qumat = QuMat(get_backend_config("qiskit"))
        qumat.create_empty_circuit(num_qubits=2)
        fm.apply(qumat, [np.pi], offset=1)
        state = np.asarray(qumat.get_final_state_vector())
        # Qiskit little-endian: qubit 1 set -> index 2.
        assert abs(state[2]) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("entanglement", "expected"),
        [
            ("full", [(0, 1), (0, 2), (1, 2)]),
            ("linear", [(0, 1), (1, 2)]),
            ("circular", [(0, 1), (1, 2), (2, 0)]),
        ],
    )
    def test_zz_pairs(self, entanglement, expected):
        assert ZZFeatureMap(3, entanglement=entanglement).pairs() == expected

    def test_zz_gate_count(self):
        # per rep: n H + n RZ + 3 gates per pair
        fm = ZZFeatureMap(num_qubits=3, reps=2)
        assert len(fm.operations(np.zeros(3))) == 2 * (3 + 3 + 3 * 3)

    def test_gateop_inverse(self):
        assert GateOp("ry", (0,), 0.5).inverse() == GateOp("ry", (0,), -0.5)
        assert GateOp("cnot", (0, 1)).inverse() == GateOp("cnot", (0, 1))


class TestValidation:
    def test_wrong_feature_count(self):
        fm = AngleFeatureMap(num_qubits=2)
        with pytest.raises(ValueError, match="expects 2 features"):
            fidelity_kernel([0.1], [0.2, 0.3], fm, get_backend_config("qiskit"))

    def test_non_finite_features(self):
        fm = AngleFeatureMap(num_qubits=1)
        with pytest.raises(ValueError, match="finite"):
            kernel_matrix(
                [[np.nan]], feature_map=fm, backend_config=get_backend_config("qiskit")
            )

    def test_unknown_method(self):
        fm = AngleFeatureMap(num_qubits=1)
        with pytest.raises(ValueError, match="method must be one of"):
            fidelity_kernel(
                [0.1], [0.2], fm, get_backend_config("qiskit"), method="magic"
            )

    def test_bad_shots(self):
        fm = AngleFeatureMap(num_qubits=1)
        with pytest.raises(ValueError, match="shots"):
            fidelity_kernel(
                [0.1],
                [0.2],
                fm,
                get_backend_config("qiskit"),
                method="swap_test",
                shots=0,
            )

    def test_bad_feature_map_args(self):
        with pytest.raises(ValueError, match="num_qubits"):
            AngleFeatureMap(num_qubits=0)
        with pytest.raises(ValueError, match="entanglement"):
            ZZFeatureMap(num_qubits=2, entanglement="star")
        with pytest.raises(ValueError, match="reps"):
            ZZFeatureMap(num_qubits=2, reps=0)

    def test_does_not_mutate_backend_config(self):
        config = get_backend_config("qiskit")
        before = {**config, "backend_options": dict(config["backend_options"])}
        fidelity_kernel(
            [0.1], [0.2], AngleFeatureMap(1), config, method="swap_test", shots=10
        )
        assert config == before


class TestKernelFromStates:
    def test_known_states(self):
        zero = [1.0, 0.0]
        one = [0.0, 1.0]
        plus = [2**-0.5, 2**-0.5]
        gram = kernel_from_states([zero, one, plus])
        expected = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]])
        np.testing.assert_allclose(gram, expected, atol=1e-12)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="equal width"):
            kernel_from_states([[1.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]])
