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

To see the full list of options and defaults, append `--help`:

```bash
uv run python benchmark/encoding_benchmarks/pennylane_baseline/iris_amplitude.py --help
uv run python benchmark/encoding_benchmarks/qdp_pipeline/iris_amplitude.py --help
```
