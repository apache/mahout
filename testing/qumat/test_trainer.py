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
"""Tests for qumat.trainer (QMLTrainer)."""

import math
import time

import numpy as np
import pytest

from qumat import (
    AngleEncodedClassifier,
    QMLModel,
    QMLTrainer,
    QuMat,
    TrainerCallback,
    TrainingArguments,
)
from qumat.optimize import canonical_state_vector, expectation_z
from qumat.trainer import binary_accuracy, mse_loss

CIRQ_CONFIG = {
    "backend_name": "cirq",
    "backend_options": {"simulator_type": "default", "shots": 1},
}


def parity_dataset(n_samples, seed):
    """Two features in [0, pi/2]; label = sign(cos 2x_0 * cos 2x_1).

    With angle encoding each qubit has <Z_k> = cos(2 x_k), so the label is
    the sign of <Z_0 Z_1>: an XOR-like problem that a single linear
    readout on one qubit cannot solve without the entangling CNOT.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, np.pi / 2, size=(n_samples, 2))
    y = np.where(np.cos(2 * x[:, 0]) * np.cos(2 * x[:, 1]) >= 0, 1.0, -1.0)
    return x, y


class TestAngleEncodedClassifier:
    def test_satisfies_protocol(self):
        assert isinstance(AngleEncodedClassifier(2), QMLModel)

    @pytest.mark.parametrize("backend_name", ["qiskit", "cirq"])
    def test_encoding_matches_qdp_angle_formula(self, backend_name):
        """RY(2x)|0> = cos(x)|0> + sin(x)|1>, QDP's angle encoding."""
        model = AngleEncodedClassifier(num_qubits=2)
        qumat = QuMat(
            {"backend_name": backend_name, "backend_options": {"shots": 1}}
            if backend_name == "cirq"
            else {
                "backend_name": "qiskit",
                "backend_options": {"simulator_type": "statevector_simulator"},
            }
        )
        qumat.create_empty_circuit(num_qubits=2)
        model.build_encoding(qumat)

        x = np.array([0.3, 1.1])
        state = canonical_state_vector(qumat, model.encode(x))

        c0, s0, c1, s1 = np.cos(x[0]), np.sin(x[0]), np.cos(x[1]), np.sin(x[1])
        # Canonical ordering is |q0 q1>.
        expected = np.array([c0 * c1, c0 * s1, s0 * c1, s0 * s1])
        np.testing.assert_allclose(state, expected, atol=1e-6)

    def test_encoding_matches_qdp_torch_reference(self):
        """Cross-check against QDP's CPU PyTorch reference encoder."""
        torch = pytest.importorskip("torch")
        torch_ref = pytest.importorskip("qumat_qdp.torch_ref")

        model = AngleEncodedClassifier(num_qubits=3)
        qumat = QuMat(
            {
                "backend_name": "qiskit",
                "backend_options": {"simulator_type": "statevector_simulator"},
            }
        )
        qumat.create_empty_circuit(num_qubits=3)
        model.build_encoding(qumat)
        x = np.array([0.3, 1.1, -0.7])

        qdp_state = torch_ref.angle_encode(
            torch.tensor(x[None, :], dtype=torch.float64), 3
        )
        # Both QDP and Qiskit put qubit 0 in the least significant bit.
        qumat.bind_parameters(model.encode(x))
        qiskit_state = np.asarray(qumat.get_final_state_vector())
        np.testing.assert_allclose(qiskit_state, qdp_state.numpy()[0], atol=1e-12)

    def test_rotation_encoding_uses_raw_angles(self):
        model = AngleEncodedClassifier(num_qubits=1, encoding="rotation")
        assert model.encode(np.array([0.4])) == {"x_0": 0.4}

    def test_constructor_errors(self):
        with pytest.raises(ValueError, match="num_qubits"):
            AngleEncodedClassifier(0)
        with pytest.raises(ValueError, match="num_layers"):
            AngleEncodedClassifier(2, num_layers=0)
        with pytest.raises(ValueError, match="unknown encoding"):
            AngleEncodedClassifier(2, encoding="amplitude")
        with pytest.raises(ValueError, match="features per sample"):
            AngleEncodedClassifier(2).encode(np.array([1.0, 2.0, 3.0]))


class TestTrainingArguments:
    def test_defaults(self):
        args = TrainingArguments()
        assert args.optimizer == "adam"
        assert args.gradient == "parameter_shift"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"epochs": 0}, "epochs"),
            ({"batch_size": 0}, "batch_size"),
            ({"log_every": -1}, "log_every"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"optimizer": "sgd-momentum"}, "unknown optimizer"),
            ({"gradient": "autograd"}, "unknown gradient"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TrainingArguments(**kwargs)


class TestQMLTrainer:
    def test_end_to_end_qiskit_statevector(self):
        """Toy binary classification on Qiskit: loss goes down, quickly."""
        x_train, y_train = parity_dataset(16, seed=7)
        x_eval, y_eval = parity_dataset(8, seed=8)
        args = TrainingArguments(epochs=5, learning_rate=0.3, batch_size=4, seed=1)

        start = time.perf_counter()
        trainer = QMLTrainer(
            AngleEncodedClassifier(num_qubits=2, num_layers=1),
            args,
            train_data=(x_train, y_train),
            eval_data=(x_eval, y_eval),
            compute_metrics=binary_accuracy,
        )
        before = trainer.evaluate()
        output = trainer.train()
        after = trainer.evaluate()
        elapsed = time.perf_counter() - start

        losses = [epoch["train_loss"] for epoch in output.history]
        assert len(losses) == 5
        assert losses[-1] < losses[0]
        assert after["eval_loss"] < before["eval_loss"]
        assert after["eval_accuracy"] >= 0.75
        # Per epoch: 16 samples x (1 forward + 2 params x 2 shifts), plus
        # one forward per eval sample for the per-epoch evaluation.
        assert output.n_circuit_evaluations == 5 * (16 * 5 + 8)
        assert elapsed < 60, f"training took {elapsed:.1f}s"
        # The trained params are bound on the trainer's circuit.
        assert all(trainer.qumat.parameters[n] is not None for n in output.params)

    def test_gradient_matches_finite_difference_of_loss(self):
        """Chain rule: d loss/d w = loss'(f) * parameter-shift d f/d w."""
        model = AngleEncodedClassifier(num_qubits=2, num_layers=2)
        args = TrainingArguments(backend_config=CIRQ_CONFIG, seed=3)
        x, y = np.array([0.2, 1.0]), -1.0
        trainer = QMLTrainer(model, args, train_data=(x[None, :], [y]))

        _, grad = trainer._sample_gradient(x, y)

        step = 1e-3
        for name in model.param_names:
            plus = dict(trainer.params, **{name: trainer.params[name] + step})
            minus = dict(trainer.params, **{name: trainer.params[name] - step})
            numeric = (
                mse_loss(model.forward(trainer.qumat, x, plus), y)
                - mse_loss(model.forward(trainer.qumat, x, minus), y)
            ) / (2 * step)
            assert grad[name] == pytest.approx(numeric, abs=1e-3), name

    def test_custom_loss_and_model(self):
        """A user-defined model and loss, trained on Cirq."""

        class OneQubitRegressor:
            num_qubits = 1
            param_names = ("w",)

            def build_circuit(self, qumat):
                qumat.apply_ry_gate(0, "x")
                qumat.apply_ry_gate(0, "w")

            def init_params(self, rng):
                return {"w": 0.0}

            def forward(self, qumat, x, params):
                return expectation_z(qumat, {"x": float(x[0]), **params})

        def absolute_error(prediction, target):
            return abs(prediction - target)

        # Targets cos(x + 0.8): the best w is 0.8.
        x = np.linspace(-1, 1, 6).reshape(-1, 1)
        y = np.cos(x[:, 0] + 0.8)
        args = TrainingArguments(
            epochs=30,
            learning_rate=0.1,
            batch_size=6,
            optimizer="gradient_descent",
            backend_config=CIRQ_CONFIG,
        )
        trainer = QMLTrainer(
            OneQubitRegressor(), args, train_data=(x, y), loss_fn=absolute_error
        )
        output = trainer.train()

        assert output.params["w"] == pytest.approx(0.8, abs=0.1)
        np.testing.assert_allclose(trainer.predict(x), y, atol=0.1)

    def test_callbacks_and_early_stop(self):
        events = []

        class Recorder(TrainerCallback):
            def on_train_begin(self, trainer):
                events.append("begin")

            def on_epoch_end(self, trainer, epoch, logs):
                events.append(("epoch", epoch, sorted(logs)))
                if epoch == 2:
                    trainer.should_stop = True

            def on_train_end(self, trainer):
                events.append("end")

        x, y = parity_dataset(4, seed=0)
        args = TrainingArguments(epochs=10, backend_config=CIRQ_CONFIG)
        trainer = QMLTrainer(
            AngleEncodedClassifier(2),
            args,
            train_data=(x, y),
            eval_data=(x, y),
            callbacks=[Recorder()],
        )
        output = trainer.train()

        assert events[0] == "begin"
        assert events[-1] == "end"
        assert [e[1] for e in events[1:-1]] == [1, 2]
        assert events[1][2] == ["epoch", "eval_loss", "train_loss"]
        assert len(output.history) == 2

    def test_seed_is_reproducible(self):
        x, y = parity_dataset(6, seed=0)
        runs = []
        for _ in range(2):
            args = TrainingArguments(epochs=2, seed=42, backend_config=CIRQ_CONFIG)
            trainer = QMLTrainer(AngleEncodedClassifier(2), args, train_data=(x, y))
            runs.append(trainer.train().params)
        assert runs[0] == runs[1]

    def test_logging(self, caplog):
        x, y = parity_dataset(4, seed=0)
        args = TrainingArguments(epochs=4, log_every=2, backend_config=CIRQ_CONFIG)
        trainer = QMLTrainer(AngleEncodedClassifier(2), args, train_data=(x, y))
        with caplog.at_level("INFO", logger="qumat.trainer"):
            trainer.train()
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 2
        assert messages[0].startswith("epoch=2, train_loss=")

    def test_predict_and_evaluate(self):
        args = TrainingArguments(backend_config=CIRQ_CONFIG)
        model = AngleEncodedClassifier(1, encoding="rotation")
        trainer = QMLTrainer(model, args)
        trainer.params = {"w_0_0": 0.0}

        predictions = trainer.predict([[0.0], [math.pi]])
        np.testing.assert_allclose(predictions, [1.0, -1.0], atol=1e-6)
        # A single sample may be passed as a 1-D array.
        assert trainer.predict([0.0]).shape == (1,)

        metrics = trainer.evaluate(([[0.0], [math.pi]], [1.0, 1.0]))
        assert metrics["eval_loss"] == pytest.approx(2.0, abs=1e-5)

    def test_errors(self):
        args = TrainingArguments(backend_config=CIRQ_CONFIG)
        model = AngleEncodedClassifier(2)

        with pytest.raises(TypeError, match="QMLModel protocol"):
            QMLTrainer(object(), args)  # type: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="unknown loss"):
            QMLTrainer(model, args, loss_fn="hinge")
        with pytest.raises(TypeError, match="pair"):
            QMLTrainer(model, args, train_data=np.zeros((3, 2)))  # type: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="must match"):
            QMLTrainer(model, args, train_data=(np.zeros((3, 2)), [1.0]))
        with pytest.raises(ValueError, match="2-D"):
            QMLTrainer(model, args, train_data=(np.zeros(3), [1.0, 1.0, 1.0]))
        with pytest.raises(ValueError, match="empty"):
            QMLTrainer(model, args, train_data=(np.zeros((0, 2)), []))

        trainer = QMLTrainer(model, args)
        with pytest.raises(ValueError, match="needs train_data"):
            trainer.train()
        with pytest.raises(ValueError, match="needs eval_data"):
            trainer.evaluate()

        class BadNames(AngleEncodedClassifier):
            @property
            def param_names(self):
                return ["not_in_circuit"]

        with pytest.raises(ValueError, match="not parameters of the built"):
            QMLTrainer(BadNames(2), args)
