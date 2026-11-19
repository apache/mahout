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
"""A small trainer for quantum machine learning models built on Qumat.

The design follows the shape of the HuggingFace ``Trainer``: a *model*
says what to compute, :class:`TrainingArguments` says how to train it,
and :class:`QMLTrainer` runs the loop (batching, gradients, optimizer
steps, evaluation, logging and callbacks)::

    model = AngleEncodedClassifier(num_qubits=2, num_layers=1)
    args = TrainingArguments(epochs=5, learning_rate=0.2, batch_size=4)
    trainer = QMLTrainer(model, args, (x_train, y_train), (x_test, y_test),
                         compute_metrics=binary_accuracy)
    trainer.train()
    trainer.evaluate()
    trainer.predict(x_test)

Gradients come from :func:`qumat.optimize.parameter_shift_gradient`
applied to the model output, combined with the derivative of the loss
by the chain rule, so only the linear (expectation-value) part is
differentiated on the quantum circuit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .optimize import (
    _make_step_optimizer,
    expectation_z,
    finite_difference_gradient,
    parameter_shift_gradient,
)
from .qumat import QuMat

logger = logging.getLogger(__name__)

Dataset = tuple[Any, Any]
LossFunction = Callable[[float, float], float]
MetricsFunction = Callable[[np.ndarray, np.ndarray], Mapping[str, float]]

DEFAULT_BACKEND_CONFIG: dict[str, Any] = {
    "backend_name": "qiskit",
    "backend_options": {"simulator_type": "statevector_simulator", "shots": 1},
}


# RY angle per unit of feature value. QDP's angle encoding maps x to
# cos(x)|0> + sin(x)|1>, which is RY(2x)|0>.
_ENCODING_SCALE = {"angle": 2.0, "rotation": 1.0}


@runtime_checkable
class QMLModel(Protocol):
    """What :class:`QMLTrainer` needs from a model.

    The circuit is built once. Input features are circuit parameters too:
    :meth:`forward` binds them for each sample together with the trainable
    parameters, so the circuit is never rebuilt during training.

    For ``gradient="parameter_shift"`` (the default), :meth:`forward` must
    return a quantity that is linear in the final state, such as an
    expectation value (optionally scaled or shifted by constants).
    """

    num_qubits: int

    @property
    def param_names(self) -> Sequence[str]:
        """Names of the trainable circuit parameters."""
        ...

    def build_circuit(self, qumat: QuMat) -> None:
        """Apply the model's gates to ``qumat``'s empty circuit."""
        ...

    def init_params(self, rng: np.random.Generator) -> dict[str, float]:
        """Return initial values for every name in :attr:`param_names`."""
        ...

    def forward(
        self, qumat: QuMat, x: np.ndarray, params: Mapping[str, float]
    ) -> float:
        """Return the model output for one sample ``x``."""
        ...


class AngleEncodedClassifier:
    """Angle encoding followed by a hardware-efficient variational ansatz.

    With ``encoding="angle"`` (the default), feature ``k`` of a sample is
    written to qubit ``k`` with ``RY(2 * x_k)``. This prepares
    ``cos(x_k)|0> + sin(x_k)|1>`` on each qubit, the same state QDP's
    ``"angle"`` encoding produces for the angles ``x`` (QDP numbers qubit 0
    as the least significant bit of its amplitude index, like Qiskit).
    With ``encoding="rotation"`` the features are used as rotation angles
    directly, ``RY(x_k)``, for data that is already scaled to angles. Each layer then applies a trainable ``RY`` to every
    qubit followed by a chain of CNOTs ``0 -> 1 -> ... -> n-1``. The output
    is ``<Z>`` on the last qubit, which the CNOT chain makes depend on every
    feature. It lies in ``[-1, 1]``, so labels should be ``-1`` / ``+1`` (or
    regression targets in that range).

    :param num_qubits: Number of qubits, equal to the number of features.
    :param num_layers: Number of trainable layers.
    :param readout_qubits: Qubits whose Z-product is the output. Defaults to
        the last qubit.
    :param encoding: ``"angle"`` (QDP semantics) or ``"rotation"``.
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 1,
        readout_qubits: Sequence[int] | None = None,
        encoding: str = "angle",
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be at least 1, got {num_qubits}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if encoding not in _ENCODING_SCALE:
            raise ValueError(
                f"unknown encoding {encoding!r}; expected one of "
                f"{sorted(_ENCODING_SCALE)}"
            )
        self.encoding = encoding
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.readout_qubits = (
            (num_qubits - 1,) if readout_qubits is None else tuple(readout_qubits)
        )
        self.feature_names = [f"x_{k}" for k in range(num_qubits)]
        self._param_names = [
            f"w_{layer}_{qubit}"
            for layer in range(num_layers)
            for qubit in range(num_qubits)
        ]

    @property
    def param_names(self) -> list[str]:
        return list(self._param_names)

    def build_encoding(self, qumat: QuMat) -> None:
        """Apply one ``RY`` per qubit, parameterized by the feature names."""
        for qubit, name in enumerate(self.feature_names):
            qumat.apply_ry_gate(qubit, name)

    def build_circuit(self, qumat: QuMat) -> None:
        self.build_encoding(qumat)
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                qumat.apply_ry_gate(qubit, f"w_{layer}_{qubit}")
            for qubit in range(self.num_qubits - 1):
                qumat.apply_cnot_gate(qubit, qubit + 1)

    def init_params(self, rng: np.random.Generator) -> dict[str, float]:
        values = rng.uniform(-np.pi, np.pi, size=len(self._param_names))
        return {name: float(v) for name, v in zip(self._param_names, values)}

    def encode(self, x: np.ndarray) -> dict[str, float]:
        """Map one sample to the values of the encoding ``RY`` angles."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.num_qubits:
            raise ValueError(
                f"expected {self.num_qubits} features per sample, got {x.size}"
            )
        scale = _ENCODING_SCALE[self.encoding]
        return {name: scale * float(v) for name, v in zip(self.feature_names, x)}

    def forward(
        self, qumat: QuMat, x: np.ndarray, params: Mapping[str, float]
    ) -> float:
        values = {**self.encode(x), **params}
        return expectation_z(qumat, values, qubits=self.readout_qubits)


@dataclass
class TrainingArguments:
    """How to train: the counterpart of HuggingFace ``TrainingArguments``.

    :ivar epochs: Passes over the training data.
    :ivar learning_rate: Optimizer step size.
    :ivar batch_size: Samples per optimizer step (gradients are averaged).
    :ivar optimizer: ``"adam"`` or ``"gradient_descent"``.
    :ivar seed: Seed for parameter initialization and shuffling.
    :ivar log_every: Log metrics every this many epochs (0 disables logging;
        the history is always recorded).
    :ivar shuffle: Shuffle the training data each epoch.
    :ivar gradient: ``"parameter_shift"`` or ``"finite_difference"``.
    :ivar backend_config: QuMat backend configuration. Defaults to the
        Qiskit statevector simulator.
    """

    epochs: int = 10
    learning_rate: float = 0.1
    batch_size: int = 8
    optimizer: str = "adam"
    seed: int | None = 0
    log_every: int = 1
    shuffle: bool = True
    gradient: str = "parameter_shift"
    backend_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.log_every < 0:
            raise ValueError(f"log_every must be >= 0, got {self.log_every}")
        if self.gradient not in ("parameter_shift", "finite_difference"):
            raise ValueError(
                f"unknown gradient {self.gradient!r}; expected "
                "'parameter_shift' or 'finite_difference'"
            )
        # Validates optimizer name and learning rate.
        _make_step_optimizer(self.optimizer, self.learning_rate)


class TrainerCallback:
    """Hooks called by :class:`QMLTrainer`. Override the ones you need.

    Set ``trainer.should_stop = True`` from a hook to stop training after
    the current epoch.
    """

    def on_train_begin(self, trainer: QMLTrainer) -> None:
        pass

    def on_epoch_end(
        self, trainer: QMLTrainer, epoch: int, logs: Mapping[str, float]
    ) -> None:
        pass

    def on_train_end(self, trainer: QMLTrainer) -> None:
        pass


@dataclass
class TrainOutput:
    """Result of :meth:`QMLTrainer.train`.

    :ivar params: Trained parameter values.
    :ivar history: One dict of metrics per epoch.
    :ivar n_circuit_evaluations: Circuit evaluations during :meth:`train`,
        including the per-epoch evaluation on ``eval_data``.
    """

    params: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)
    n_circuit_evaluations: int = 0


def mse_loss(prediction: float, target: float) -> float:
    """Squared error ``(prediction - target) ** 2``."""
    return (prediction - target) ** 2


def binary_accuracy(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Accuracy for ``-1``/``+1`` labels, thresholding predictions at 0."""
    predicted = np.where(np.asarray(predictions) >= 0, 1.0, -1.0)
    return {"accuracy": float(np.mean(predicted == np.asarray(targets)))}


_LOSSES: dict[str, tuple[LossFunction, Callable[[float, float], float]]] = {
    "mse": (mse_loss, lambda p, t: 2.0 * (p - t)),
}


def _loss_derivative(loss: LossFunction) -> Callable[[float, float], float]:
    """d loss / d prediction by central differences.

    This is purely classical arithmetic in float64, so it costs no circuit
    evaluations and a small step is accurate.
    """
    step = 1e-6

    def derivative(prediction: float, target: float) -> float:
        return (loss(prediction + step, target) - loss(prediction - step, target)) / (
            2 * step
        )

    return derivative


def _as_dataset(data: Dataset, name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        features, targets = data
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a (features, targets) pair") from exc
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"{name} features must be 2-D (samples, features)")
    if len(x) != len(y):
        raise ValueError(
            f"{name} has {len(x)} samples but {len(y)} targets; they must match"
        )
    if len(x) == 0:
        raise ValueError(f"{name} is empty")
    return x, y


class QMLTrainer:
    """Train a :class:`QMLModel` with parameter-shift gradients.

    :param model: The model to train.
    :param args: Training settings.
    :param train_data: ``(features, targets)``; features have shape
        ``(n_samples, n_features)``. NumPy arrays, lists and CPU tensors work.
    :param eval_data: Optional ``(features, targets)`` evaluated after each
        epoch and by :meth:`evaluate`.
    :param loss_fn: ``"mse"`` or a callable ``loss(prediction, target)``.
    :param compute_metrics: Optional ``f(predictions, targets) -> dict`` of
        extra evaluation metrics, e.g. :func:`binary_accuracy`.
    :param callbacks: :class:`TrainerCallback` instances.
    """

    def __init__(
        self,
        model: QMLModel,
        args: TrainingArguments | None = None,
        train_data: Dataset | None = None,
        eval_data: Dataset | None = None,
        loss_fn: str | LossFunction = "mse",
        compute_metrics: MetricsFunction | None = None,
        callbacks: Sequence[TrainerCallback] = (),
    ) -> None:
        if not isinstance(model, QMLModel):
            raise TypeError(
                "model must implement the QMLModel protocol (num_qubits, "
                "param_names, build_circuit, init_params, forward)"
            )
        self.model = model
        self.args = args if args is not None else TrainingArguments()
        self.train_data = (
            _as_dataset(train_data, "train_data") if train_data is not None else None
        )
        self.eval_data = (
            _as_dataset(eval_data, "eval_data") if eval_data is not None else None
        )
        if isinstance(loss_fn, str):
            if loss_fn not in _LOSSES:
                raise ValueError(
                    f"unknown loss {loss_fn!r}; expected one of {sorted(_LOSSES)} "
                    "or a callable"
                )
            self.loss_fn, self._loss_grad = _LOSSES[loss_fn]
        else:
            self.loss_fn = loss_fn
            self._loss_grad = _loss_derivative(loss_fn)
        self.compute_metrics = compute_metrics
        self.callbacks = list(callbacks)

        self.qumat = QuMat(dict(self.args.backend_config or DEFAULT_BACKEND_CONFIG))
        self.qumat.create_empty_circuit(num_qubits=model.num_qubits)
        model.build_circuit(self.qumat)
        missing = [n for n in model.param_names if n not in self.qumat.parameters]
        if missing:
            raise ValueError(
                f"model.param_names {missing} are not parameters of the built circuit"
            )

        self._rng = np.random.default_rng(self.args.seed)
        self.params: dict[str, float] = dict(model.init_params(self._rng))
        self.history: list[dict[str, float]] = []
        self.should_stop = False
        self.n_circuit_evaluations = 0

    # -- internals -----------------------------------------------------

    def _forward(
        self, qumat: QuMat, x: np.ndarray, params: Mapping[str, float]
    ) -> float:
        self.n_circuit_evaluations += 1
        return float(self.model.forward(qumat, x, params))

    def _sample_gradient(
        self, x: np.ndarray, target: float
    ) -> tuple[float, dict[str, float]]:
        """Loss and d loss / d params for one sample (chain rule)."""
        names = list(self.model.param_names)
        prediction = self._forward(self.qumat, x, self.params)
        grad_fn = (
            parameter_shift_gradient
            if self.args.gradient == "parameter_shift"
            else finite_difference_gradient
        )
        output_grad = grad_fn(
            self.qumat,
            lambda qm, p: self._forward(qm, x, p),
            self.params,
            names,
        )
        scale = self._loss_grad(prediction, target)
        loss = float(self.loss_fn(prediction, target))
        return loss, {name: scale * output_grad[name] for name in names}

    def _log(self, logs: Mapping[str, float]) -> None:
        epoch = int(logs["epoch"])
        if self.args.log_every and epoch % self.args.log_every == 0:
            text = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in logs.items()
            )
            logger.info("%s", text)

    # -- public API ----------------------------------------------------

    def train(self) -> TrainOutput:
        """Run the training loop and return the trained parameters."""
        if self.train_data is None:
            raise ValueError("train() needs train_data")
        x_train, y_train = self.train_data
        names = list(self.model.param_names)
        optimizer = _make_step_optimizer(self.args.optimizer, self.args.learning_rate)
        self.should_stop = False
        evaluations_before = self.n_circuit_evaluations

        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(1, self.args.epochs + 1):
            order = (
                self._rng.permutation(len(x_train))
                if self.args.shuffle
                else np.arange(len(x_train))
            )
            losses = []
            for start in range(0, len(order), self.args.batch_size):
                batch = order[start : start + self.args.batch_size]
                total = np.zeros(len(names))
                for index in batch:
                    loss, grad = self._sample_gradient(x_train[index], y_train[index])
                    losses.append(loss)
                    total += np.array([grad[name] for name in names])
                x = np.array([self.params[name] for name in names])
                x = optimizer.step(x, total / len(batch))
                self.params = {name: float(v) for name, v in zip(names, x)}

            logs: dict[str, float] = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
            }
            if self.eval_data is not None:
                logs.update(self.evaluate())
            self.history.append(logs)
            self._log(logs)
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, logs)
            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end(self)
        self.qumat.bind_parameters(self.params)
        return TrainOutput(
            params=dict(self.params),
            history=list(self.history),
            n_circuit_evaluations=self.n_circuit_evaluations - evaluations_before,
        )

    def predict(self, features: Any) -> np.ndarray:
        """Model outputs for each row of ``features`` with current params."""
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.array([self._forward(self.qumat, row, self.params) for row in x])

    def evaluate(self, eval_data: Dataset | None = None) -> dict[str, float]:
        """Mean loss (``eval_loss``) and ``compute_metrics`` on ``eval_data``.

        Uses the ``eval_data`` given to the constructor when omitted. Metric
        names are prefixed with ``eval_`` as in HuggingFace.
        """
        data = (
            _as_dataset(eval_data, "eval_data")
            if eval_data is not None
            else self.eval_data
        )
        if data is None:
            raise ValueError("evaluate() needs eval_data")
        x, y = data
        predictions = self.predict(x)
        metrics = {
            "eval_loss": float(
                np.mean([self.loss_fn(p, t) for p, t in zip(predictions, y)])
            )
        }
        if self.compute_metrics is not None:
            for key, value in self.compute_metrics(predictions, y).items():
                metrics[f"eval_{key}"] = float(value)
        return metrics
