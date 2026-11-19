---
title: Training QML Models (QMLTrainer)
sidebar_label: QML Trainer
---

# Training QML Models with `QMLTrainer`

`qumat.QMLTrainer` is a small training loop for quantum machine learning
models built from parameterized Qumat circuits. Its shape follows the
HuggingFace `Trainer`:

| Piece | Role |
|---|---|
| a model (`QMLModel`) | what to compute: circuit, parameters, output |
| `TrainingArguments` | how to train: epochs, learning rate, batch size, ... |
| `QMLTrainer` | the loop: batching, gradients, optimizer, evaluation, logs |

:::note Status
This is an early, deliberately small API (issue
[#1409](https://github.com/apache/mahout/issues/1409)). It runs on CPU
simulators and is meant for small circuits and experiments.
:::

## Quick start

```python
import numpy as np
from qumat import AngleEncodedClassifier, QMLTrainer, TrainingArguments
from qumat.trainer import binary_accuracy

# Toy data: 2 features in [0, pi/2], labels -1 / +1
rng = np.random.default_rng(7)
x = rng.uniform(0, np.pi / 2, size=(24, 2))
y = np.where(np.cos(2 * x[:, 0]) * np.cos(2 * x[:, 1]) >= 0, 1.0, -1.0)

model = AngleEncodedClassifier(num_qubits=2, num_layers=1)
args = TrainingArguments(epochs=5, learning_rate=0.3, batch_size=4, seed=1)

trainer = QMLTrainer(
    model,
    args,
    train_data=(x[:16], y[:16]),
    eval_data=(x[16:], y[16:]),
    compute_metrics=binary_accuracy,
)
output = trainer.train()
print(output.history[-1])        # {'epoch': 5, 'train_loss': ..., 'eval_accuracy': ...}
print(trainer.evaluate())        # {'eval_loss': ..., 'eval_accuracy': ...}
print(trainer.predict(x[16:]))   # model outputs in [-1, 1]
```

By default the trainer uses the Qiskit statevector simulator. Pass
`backend_config` in `TrainingArguments` to use another backend, for
example `{"backend_name": "cirq", "backend_options": {}}`.

## Writing your own model

Any object with these members works (it is a `typing.Protocol`, so no
base class is required):

```python
class MyModel:
    num_qubits = 1
    param_names = ("w",)

    def build_circuit(self, qumat):
        qumat.apply_ry_gate(0, "x")   # input feature, rebound per sample
        qumat.apply_ry_gate(0, "w")   # trainable weight

    def init_params(self, rng):
        return {"w": 0.0}

    def forward(self, qumat, x, params):
        from qumat import expectation_z
        return expectation_z(qumat, {"x": float(x[0]), **params})
```

The circuit is built **once**. Input features are ordinary circuit
parameters that `forward` binds for every sample, together with the
trainable parameters.

## How gradients are computed

`forward` returns an expectation value. Its gradient with respect to the
weights comes from the parameter-shift rule
(`qumat.parameter_shift_gradient`), which is exact for expectation
values. The loss, for example the squared error, is not linear in the
state, so the trainer applies it with the chain rule:

```text
d loss / d w = loss'(f(x; w), y) * d f(x; w) / d w
```

Each sample therefore costs one forward evaluation plus two evaluations
per trainable parameter (more if a parameter is reused by several gates).
`loss_fn` may be `"mse"` or any callable `loss(prediction, target)`.

## Data encoding and QDP

`AngleEncodedClassifier` writes feature `x_k` to qubit `k` with
`RY(2 * x_k)`:

```text
RY(2x)|0> = cos(x)|0> + sin(x)|1>
```

That is exactly the state QDP's `"angle"` encoding produces. QDP also
orders qubits like Qiskit (qubit 0 is the least significant bit of the
amplitude index), and the tests compare the two state vectors directly.
Use `encoding="rotation"` if your features are already rotation angles.

QDP itself runs on GPUs and is not used by the trainer. Today the
trainer accepts NumPy arrays, lists or CPU tensors of raw features and
encodes them with circuit gates. A future QDP path could hand encoded
state vectors to a simulator through DLPack instead; because the
encodings match, models trained either way would be interchangeable.

## Callbacks and logging

Subclass `TrainerCallback` and override `on_train_begin`,
`on_epoch_end(trainer, epoch, logs)` or `on_train_end`. Setting
`trainer.should_stop = True` in a callback stops after the current
epoch. Metrics are logged to the `qumat.trainer` logger every
`log_every` epochs:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## `TrainingArguments`

| Field | Default | Meaning |
|---|---|---|
| `epochs` | 10 | passes over the training data |
| `learning_rate` | 0.1 | optimizer step size |
| `batch_size` | 8 | samples per step; gradients are averaged |
| `optimizer` | `"adam"` | or `"gradient_descent"` |
| `seed` | 0 | parameter initialization and shuffling |
| `log_every` | 1 | log every N epochs (0 disables) |
| `shuffle` | `True` | reshuffle the training data each epoch |
| `gradient` | `"parameter_shift"` | or `"finite_difference"` |
| `backend_config` | Qiskit statevector | any QuMat backend configuration |
