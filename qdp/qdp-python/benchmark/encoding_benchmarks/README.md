# Encoding benchmarks

This directory is used to **compare a pure PennyLane baseline with a QDP pipeline on the full training loop**.
Both scripts use the same dataset and the same variational model; **only the encoding step is different**.

- **`pennylane_baseline/`**: pure PennyLane (sklearn / official Iris file → encoding → variational classifier).
- **`qdp_pipeline/`**: same data and model, but encoding is done via the QDP `QuantumDataLoader` (amplitude).

Run all commands from the `qdp-python` directory:

```bash
cd qdp/qdp-python
```

## Environment (one-time setup)

```bash
uv sync --group benchmark        # install PennyLane, torch(+CUDA), scikit-learn, etc.
uv run maturin develop           # build QDP Python extension (qumat_qdp)
```

If your CUDA driver is not 12.6, adjust the PyTorch index URL in `pyproject.toml` according to the comments there.

## Iris amplitude baseline (pure PennyLane)

Pipeline: 2-class Iris (sklearn or official file) → L2 normalize → `get_angles` → variational classifier.

```bash
uv run python benchmark/encoding_benchmarks/pennylane_baseline/iris_amplitude.py
```

Common flags (only the key ones):

- `--iters`: optimizer steps (default: 1500)
- `--layers`: number of variational layers (default: 10)
- `--lr`: learning rate (default: 0.08)
- `--optimizer`: `adam` or `nesterov` (default: `adam`)
- `--trials`: number of restarts; best test accuracy reported (default: 20)
- `--data-file`: use the official Iris file instead of sklearn (2 features, 75% train)

Example (official file + Nesterov + short run):

```bash
uv run python benchmark/encoding_benchmarks/pennylane_baseline/iris_amplitude.py \
  --data-file benchmark/encoding_benchmarks/pennylane_baseline/data/iris_classes1and2_scaled.txt \
  --optimizer nesterov --lr 0.01 --layers 6 --trials 3 --iters 80 --early-stop 0
```

## Iris amplitude (QDP pipeline)

Pipeline is identical to the baseline except for encoding:
4-D vectors → QDP `QuantumDataLoader` (amplitude) → `StatePrep(state_vector)` → same variational classifier.

```bash
uv run python benchmark/encoding_benchmarks/qdp_pipeline/iris_amplitude.py
```

The CLI mirrors the baseline, plus:

- **QDP-specific flags**
  - `--device-id`: QDP device id (default: 0)
  - `--data-dir`: directory for temporary `.npy` files (default: system temp directory)

Example (same settings as the baseline example, but with QDP encoding):

```bash
uv run python benchmark/encoding_benchmarks/qdp_pipeline/iris_amplitude.py \
  --data-file benchmark/encoding_benchmarks/pennylane_baseline/data/iris_classes1and2_scaled.txt \
  --optimizer nesterov --lr 0.01 --layers 6 --trials 3 --iters 80 --early-stop 0
```

## MNIST amplitude baseline (pure PennyLane)

Pipeline: 2-class MNIST (default: digits 3 vs 6) → PCA (784 → 2^qubits) → L2 normalize → `AmplitudeEmbedding` → variational classifier.

```bash
uv run python benchmark/encoding_benchmarks/pennylane_baseline/mnist_amplitude.py
```

Common flags (only the key ones):

- `--qubits`: number of qubits; PCA reduces to 2^qubits features (default: 4 → 16-D)
- `--digits`: two digits for binary classification (default: `3,6`)
- `--n-samples`: max samples per class (default: 500)
- `--iters`: optimizer steps (default: 2000)
- `--layers`: number of variational layers (default: 10)
- `--lr`: learning rate (default: 0.05)
- `--optimizer`: `adam` or `nesterov` (default: `adam`)
- `--trials`: number of restarts; best test accuracy reported (default: 10)

Example (quick test):

```bash
uv run python benchmark/encoding_benchmarks/pennylane_baseline/mnist_amplitude.py \
  --digits "3,6" --n-samples 100 --trials 3 --iters 500 --early-stop 0
```

## MNIST amplitude (QDP pipeline)

Pipeline is identical to the baseline except for encoding:
PCA-reduced vectors → QDP `QdpEngine.encode` (amplitude) → `StatePrep(state_vector)` → same variational classifier.

```bash
uv run python benchmark/encoding_benchmarks/qdp_pipeline/mnist_amplitude.py
```

The CLI mirrors the baseline, plus:

- **QDP-specific flags**
  - `--device-id`: QDP device id (default: 0)
  - `--data-dir`: directory for temporary `.npy` files (default: system temp directory)

Example (same settings as the baseline example, but with QDP encoding):

```bash
uv run python benchmark/encoding_benchmarks/qdp_pipeline/mnist_amplitude.py \
  --digits "3,6" --n-samples 100 --trials 3 --iters 500 --early-stop 0
```

## SVHN IQP baseline (pure PennyLane)

Pipeline: 2-class SVHN (digit 1 vs 7) → PCA (3072 → n_qubits) → custom IQP circuit (H^n · Diag · H^n) → variational classifier.

```bash
uv run python benchmark/encoding_benchmarks/pennylane_baseline/svhn_iqp.py
```

Common flags (only the key ones):

- `--n-qubits`: number of qubits / PCA components (default: 10)
- `--n-samples`: total samples after binary filter + subsample (default: 500)
- `--iters`: optimizer steps (default: 500)
- `--batch-size`: batch size (default: 10)
- `--layers`: number of variational layers (default: 6)
- `--lr`: learning rate (default: 0.01)
- `--optimizer`: `adam` or `nesterov` (default: `adam`)
- `--trials`: number of restarts; best test accuracy reported (default: 3)
- `--early-stop`: stop when test accuracy >= this; 0 = off (default: 0.85)
- `--backend`: `cpu` (default.qubit) or `gpu` (lightning.gpu)

Example (quick test):

```bash
uv run python benchmark/encoding_benchmarks/pennylane_baseline/svhn_iqp.py \
  --n-qubits 6 --n-samples 200 --iters 200 --trials 1 --early-stop 0 --backend cpu
```

## SVHN IQP (QDP pipeline)

Pipeline is identical to the baseline except for encoding:
PCA-reduced vectors → QDP `QdpEngine.encode(method="iqp")` (one-shot, GPU) → `StatePrep(state_vector)` → same variational classifier.

```bash
uv run python benchmark/encoding_benchmarks/qdp_pipeline/svhn_iqp.py
```

The CLI mirrors the baseline, plus:

- **QDP-specific flags**
  - `--device-id`: CUDA device id (default: 0)

Example (same settings as the baseline example, but with QDP encoding):

```bash
uv run python benchmark/encoding_benchmarks/qdp_pipeline/svhn_iqp.py \
  --n-qubits 6 --n-samples 200 --iters 200 --trials 1 --early-stop 0 --backend cpu
```

## SVHN IQP experiment runner

Run all qubit-scaling, accuracy-parity, and sample-scaling experiments:

```bash
bash benchmark/encoding_benchmarks/run_svhn_iqp_experiments.sh
```

Logs are saved to `benchmark/encoding_benchmarks/logs/`. Results and analysis are in `benchmark/encoding_benchmarks/report.md`.

## Full help

To see the full list of options and defaults, append `--help`:

```bash
uv run python benchmark/encoding_benchmarks/pennylane_baseline/iris_amplitude.py --help
uv run python benchmark/encoding_benchmarks/pennylane_baseline/mnist_amplitude.py --help
uv run python benchmark/encoding_benchmarks/pennylane_baseline/svhn_iqp.py --help
uv run python benchmark/encoding_benchmarks/qdp_pipeline/iris_amplitude.py --help
uv run python benchmark/encoding_benchmarks/qdp_pipeline/mnist_amplitude.py --help
uv run python benchmark/encoding_benchmarks/qdp_pipeline/svhn_iqp.py --help
```
