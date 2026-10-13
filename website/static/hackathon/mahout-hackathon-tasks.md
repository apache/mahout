# Apache Mahout Hackathon Tasks (3-hour format)

**Schedule**

| Time | What |
|---|---|
| 0:00 – 0:20 | Intro talk, `pip install qumat`, open the Colab notebook (Parts 0 to 3 only, 10 minutes) |
| 0:20 – 0:30 | Pick a task below, comment on the issue to claim it, fork and branch |
| 0:30 – 2:30 | Hack. Mentors in the room; one has a GPU box for QDP verification |
| 2:30 – 3:00 | Show and tell: open PRs, demo, next steps |

Every task below is sized for **one to two hours** and needs **no GPU**. Each one ends in a PR with a test. Anything bigger lives in the appendix for after the event.

Setup for every task: `git clone` your fork, `pip install uv`, `uv sync`, `uv run pre-commit install`, `uv run pytest testing/qumat` (about a minute).

---

## Track A: Qumat (Python only)

All items reproduced on 2026-10-09 against `main` (commit `bbce3c392`) and PyPI `qumat` 0.6.0. Code: `qumat/qumat.py` and `qumat/{qiskit,cirq,amazon_braket}_backend.py`. Tests: `testing/qumat/`. Each task has a GitHub issue; comment on it to claim the task.

### A1 Clear error for an unknown backend name ([#1506](https://github.com/apache/mahout/issues/1506))

```python
QuMat({"backend_name": "pennylane", "backend_options": {}})
# ModuleNotFoundError: No module named 'qumat.pennylane_backend'
```

`QuMat.__init__` builds the module path from `backend_name` and lets the import error leak. Raise a `ValueError` listing the supported backends (`qiskit`, `cirq`, `amazon_braket`) and add a test.

### A2 Unsupported `simulator_type` should raise, not `print()` ([#1507](https://github.com/apache/mahout/issues/1507))

```python
QuMat({"backend_name": "cirq", "backend_options": {"simulator_type": "density_matrix"}})
# prints "Simulator type 'density_matrix' is not supported in Cirq. Ignoring this argument" and runs anyway

QuMat({"backend_name": "amazon_braket", "backend_options": {"simulator_type": "sv1"}})
# prints "... Using default.", falls back to the AWS SV1 device, then fails with NoRegionError
```

`cirq_backend.py:34` and `amazon_braket_backend.py:51` `print()` and continue. Replace with a `ValueError` (or `warnings.warn`), add tests for the error path.

### A3 Cirq backend ignores the initialized simulator ([#1508](https://github.com/apache/mahout/issues/1508))

`cirq_backend.initialize_backend` returns a simulator, but `execute_circuit` and `get_final_state_vector` create a fresh `cirq.Simulator()` and never use the `backend` argument. Make the backend object the one that runs the circuit. (Density-matrix support is a follow-up, see appendix.)

### A4 Bug: reusing a parameter name on two gates fails on Qiskit only ([#1509](https://github.com/apache/mahout/issues/1509))

```python
qc = QuMat(qiskit_config); qc.create_empty_circuit(2)
qc.apply_rx_gate(0, "t"); qc.apply_rx_gate(1, "t")
# qiskit:        CircuitError: "name conflict adding parameter 't'"
# cirq:          ok  [Counter({3: 100})]
# amazon_braket: ok  Counter({'11': 100})
```

Cache one `qiskit.circuit.Parameter` per name in the Qiskit backend and add a cross-backend test for shared parameters.

### A5 `create_empty_circuit()` without `num_qubits` behaves differently per backend ([#1510](https://github.com/apache/mahout/issues/1510))

```python
qc.create_empty_circuit(); qc.apply_hadamard_gate(0)
# qiskit:        CircuitError: 'Index 0 out of range for size 0.'
# cirq / braket: ok, qubit allocated on demand
```

Pick one behavior (a clear `ValueError` from `QuMat` when `num_qubits` is missing is the quick option) and make all three backends match, with tests.

### A6 Add one missing gate (one gate per person) ([#1511](https://github.com/apache/mahout/issues/1511))

`QuMat` exposes exactly `not, hadamard, cnot, toffoli, swap, cswap, pauli_x, pauli_y, pauli_z, t, rx, ry, rz, u`. Pick **one** of: **S**, **S-dagger**, **T-dagger**, **CZ**, **Phase P(λ)**. Add `QuMat.apply_<gate>_gate`, implement it in all three backends (copy the pattern of `apply_t_gate`), add a row to `docs/qumat/basic-gates.md`, and a test in `testing/qumat/test_single_qubit_gates.py` or `test_multi_qubit_gates.py` that checks the state vector on all backends.

### A7 Python 3.13 support ([#1512](https://github.com/apache/mahout/issues/1512))

Lift the `<3.13` cap in `pyproject.toml` and add 3.13 to the CI matrix. All dependencies already ship 3.13 wheels.

---

## Track B: QDP

### Python only, no Rust toolchain needed

| # | Issue | Task | Where |
|---|---|---|---|
| B1 | [#1469](https://github.com/apache/mahout/issues/1469) | `QdpBenchmark` does not validate builder argument ranges (zero batches, zero qubits, negative warmup are accepted) | `qdp/qdp-python/qumat_qdp/`, pure Python; tests in `testing/qdp_python/` run without the extension |
| B2 | [#1331](https://github.com/apache/mahout/issues/1331) | QDP tests instantiate `QdpEngine` or use `device="cuda"` without a GPU check | `testing/qdp/`; add `requires_qdp` / `pytest.mark.skipif` so the suite skips instead of erroring |
| B3 | [#1468](https://github.com/apache/mahout/issues/1468) | Rust-backed file loader ignores the configured batch limit | one-line fix at `qumat_qdp/loader.py:620` (`batch_limit=None`) plus a unit test with a fake engine, like `testing/qdp_python/test_fallback.py`; a mentor verifies on the GPU box |

### Rust, only if you already have `cargo` installed

The first `cargo build` of `qdp-core` pulls arrow, parquet and friends and takes 5 to 10 minutes on a laptop, so start it right after claiming. No CUDA needed: without `nvcc` the build uses stubs (`QDP_NO_CUDA=1 cargo test -p qdp-core --test <name>`), exactly like CI.

| # | Issue | Task | Where |
|---|---|---|---|
| B4 | [#1472](https://github.com/apache/mahout/issues/1472) | QDP cannot read default LZ4-compressed Feather files | `qdp-core/src/readers/arrow_ipc.rs`; enable Arrow's `ipc_compression` feature in `qdp/Cargo.toml`, add an LZ4 fixture test to `qdp-core/tests/arrow_ipc_io.rs` |
| B5 | [#787](https://github.com/apache/mahout/issues/787) | NumPy file size check to prevent OOM | `qdp-core/src/readers/numpy.rs`; check the file size against the expected shape before reading, test in `qdp-core/tests/numpy.rs` |
| B6 | [#1436](https://github.com/apache/mahout/issues/1436) | `StreamingProducer`: use `VecDeque` for O(1) buffer advance | `qdp-core/src/pipeline_runner.rs`; the StreamingProducer inline tests run on CPU |

---

## Appendix: follow-ups after the hackathon

Half-day and larger items, kept here so teams that finish early or want to continue afterwards can pick them up.

**Qumat**

- Density-matrix simulation on Cirq (`simulator_type="density_matrix"`), builds on A3.
- Controlled rotations CRx, CRy, CRz and RZZ accepting a float or a parameter name.
- Unified result format and bit ordering: today X on qubit 0 gives `{'01': 10}` on Qiskit, `[Counter({2: 10})]` on Cirq, `Counter({'10': 10})` on Braket.
- `QuMat.measure(qubits)`; `examples/qumat/quantum_teleportation.py` currently calls `qc.circuit.measure` directly, Qiskit only.
- `QuMat.expectation_value(observable)` for Pauli strings, then parameter-shift gradients.
- OpenQASM export, batch parameter sweeps, ansatz templates, `initialize_state(vector)` as the QDP-to-Qumat bridge.

**QDP, fix on a laptop, verify once on a GPU**

- [#1471](https://github.com/apache/mahout/issues/1471) `float32_pipeline` keyword mismatch in `qdp-python/src/lib.rs:36`
- [#1467](https://github.com/apache/mahout/issues/1467) `.pt`/`.pth` loading: enable the existing `pytorch` feature in the wheel build (needs libtorch)
- [#1434](https://github.com/apache/mahout/issues/1434) `Source` enum for `PipelineIterator`
- [#1430](https://github.com/apache/mahout/issues/1430) fail fast when `estimate_memory()` exceeds free VRAM
- [#1431](https://github.com/apache/mahout/issues/1431) PyO3 bindings for `Preprocessor`, then [#1432](https://github.com/apache/mahout/issues/1432) docs
- [#1433](https://github.com/apache/mahout/issues/1433) `KernelElem` trait for the f32/f64 FFI stubs

**QDP, needs CUDA kernel work**

- [#1008](https://github.com/apache/mahout/issues/1008) ZZFeatureMap encoding
- [#1370](https://github.com/apache/mahout/issues/1370) drop mid-stream D2H norm validation in batch encoding

**Building QDP from source with a GPU** (your own Linux box or a Colab T4; first build 10 to 20 minutes on Colab):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export QDP_CUDA_ARCH_LIST=75          # T4; omit to build every supported architecture
pip install --no-deps . && pip install ./qdp/qdp-python
python -m pytest testing/qdp testing/qdp_python -q
```
