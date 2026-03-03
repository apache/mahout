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

"""tests for Quantum Data Loader."""

from unittest.mock import patch

import numpy as np
import pytest

try:
    from qumat_qdp.loader import QuantumDataLoader
except ImportError:
    QuantumDataLoader = None  # type: ignore[assignment,misc]


def _loader_available():
    return QuantumDataLoader is not None


def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_mutual_exclusion_both_sources_raises() -> None:
    """Calling both .source_synthetic() and .source_file() then __iter__ raises ValueError."""
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(4)
        .batches(10, size=4)
        .source_synthetic()
        .source_file("/tmp/any.parquet")
    )
    with pytest.raises(ValueError) as exc_info:
        list(loader)
    msg = str(exc_info.value)
    assert "Cannot set both synthetic and file sources" in msg
    assert "source_synthetic" in msg
    assert "source_file" in msg


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_mutual_exclusion_exact_message() -> None:
    """ValueError when both sources set: message mentions source_synthetic and source_file."""
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(4)
        .batches(10, size=4)
        .source_file("/tmp/x.npy")
        .source_synthetic()
    )
    with pytest.raises(ValueError) as exc_info:
        list(loader)
    assert "Cannot set both synthetic and file sources" in str(exc_info.value)


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_source_file_empty_path_raises() -> None:
    """source_file() with empty path raises ValueError."""
    loader = QuantumDataLoader(device_id=0).qubits(4).batches(10, size=4)
    with pytest.raises(ValueError) as exc_info:
        loader.source_file("")
    assert "path" in str(exc_info.value).lower()


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_synthetic_loader_batch_count() -> None:
    """Synthetic loader yields exactly total_batches batches."""
    total = 5
    batch_size = 4
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(4)
        .batches(total, size=batch_size)
        .source_synthetic()
    )
    try:
        batches = list(loader)
    except RuntimeError as e:
        if "only available on Linux" in str(e) or "not available" in str(e):
            pytest.skip("CUDA/Linux required for loader iteration")
        raise
    assert len(batches) == total


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_file_loader_unsupported_extension_raises() -> None:
    """source_file with unsupported extension raises at __iter__."""
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(4)
        .batches(10, size=4)
        .source_file("/tmp/data.unsupported")
    )
    try:
        list(loader)
    except RuntimeError as e:
        msg = str(e).lower()
        if "not available" in msg:
            pytest.skip(
                "create_file_loader not available (e.g. extension built without loader)"
            )
            return
        assert "unsupported" in msg or "extension" in msg or "supported" in msg
        return
    except ValueError:
        pytest.skip("Loader may validate path before Rust")
        return
    pytest.fail("Expected RuntimeError for unsupported file extension")


# --- Streaming (source_file(..., streaming=True)) tests ---


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_streaming_requires_parquet() -> None:
    """source_file(path, streaming=True) with non-.parquet path raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        QuantumDataLoader(device_id=0).qubits(4).batches(10, size=4).source_file(
            "/tmp/data.npy", streaming=True
        )
    msg = str(exc_info.value).lower()
    assert "parquet" in msg or "streaming" in msg


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_streaming_parquet_extension_ok() -> None:
    """source_file(path, streaming=True) with .parquet path does not raise at builder."""
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(4)
        .batches(10, size=4)
        .source_file("/tmp/data.parquet", streaming=True)
    )
    # Iteration may raise RuntimeError (no CUDA) or fail on missing file; we only check builder accepts.
    assert loader._streaming_requested is True
    assert loader._file_path == "/tmp/data.parquet"


# --- NullHandling builder tests ---


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_null_handling_fill_zero() -> None:
    """null_handling('fill_zero') sets the field correctly."""
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(4)
        .batches(10, size=4)
        .null_handling("fill_zero")
    )
    assert loader._null_handling == "fill_zero"


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_null_handling_reject() -> None:
    """null_handling('reject') sets the field correctly."""
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(4)
        .batches(10, size=4)
        .null_handling("reject")
    )
    assert loader._null_handling == "reject"


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_null_handling_invalid_raises() -> None:
    """null_handling with an invalid string raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        QuantumDataLoader(device_id=0).null_handling("invalid_policy")
    msg = str(exc_info.value)
    assert "fill_zero" in msg or "reject" in msg


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_null_handling_default_is_none() -> None:
    """By default, _null_handling is None (Rust will use FillZero)."""
    loader = QuantumDataLoader(device_id=0)
    assert loader._null_handling is None


# --- S3 URL (source_file) builder tests ---


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
@pytest.mark.parametrize(
    ("path", "streaming"),
    [
        ("s3://my-bucket/data.parquet", False),
        ("s3://bucket/path/to/data.parquet", True),
        ("s3://bucket/data.parquet?versionId=abc123", False),
        ("s3://bucket/data.parquet?versionId=abc123", True),
        ("s3://bucket/data.npy", False),
    ],
    ids=[
        "parquet-no-stream",
        "parquet-stream",
        "parquet-query-no-stream",
        "parquet-query-stream",
        "npy-no-stream",
    ],
)
def test_source_file_s3_accepted(path, streaming):
    """source_file() accepts valid S3 URLs at builder level."""
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(4)
        .batches(10, size=4)
        .source_file(path, streaming=streaming)
    )
    assert loader._file_path == path
    assert loader._file_requested is True
    assert loader._streaming_requested is streaming


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
@pytest.mark.parametrize(
    "path",
    [
        "s3://bucket/data.npy",
        "s3://bucket/data.npy?versionId=abc",
    ],
    ids=["npy", "npy-query"],
)
def test_source_file_s3_streaming_non_parquet_raises(path):
    """source_file(s3://..., streaming=True) with non-.parquet raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        QuantumDataLoader(device_id=0).qubits(4).batches(10, size=4).source_file(
            path, streaming=True
        )
    msg = str(exc_info.value).lower()
    assert "parquet" in msg or "streaming" in msg


# --- as_torch() / as_numpy() output format tests ---


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_as_torch_raises_at_config_time_when_torch_missing():
    """as_torch() raises RuntimeError immediately (config time) when torch is not installed."""
    with patch("qumat_qdp.loader._torch", None):
        loader = QuantumDataLoader(device_id=0).qubits(4).batches(2, size=4)
        with pytest.raises(RuntimeError) as exc_info:
            loader.as_torch()
        msg = str(exc_info.value)
        assert "PyTorch" in msg or "torch" in msg.lower()
        assert "pip install" in msg


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_as_numpy_succeeds_at_config_time_without_torch():
    """as_numpy() does not raise at config time even when torch is not installed."""
    with patch("qumat_qdp.loader._torch", None):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(4)
            .batches(2, size=4)
            .source_synthetic()
            .as_numpy()
        )
    assert loader._output_format == ("numpy",)


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_as_numpy_yields_float64_arrays():
    """as_numpy() yields numpy float64 arrays with correct shape; no torch required."""
    num_qubits = 4
    batch_size = 8
    state_len = 2**num_qubits  # 16

    batches = []
    with patch("qumat_qdp.loader._torch", None):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(num_qubits)
            .batches(3, size=batch_size)
            .source_synthetic()
            .as_numpy()
        )
        for batch in loader:
            batches.append(batch)

    assert len(batches) == 3
    for batch in batches:
        assert isinstance(batch, np.ndarray), f"expected ndarray, got {type(batch)}"
        assert batch.dtype == np.float64, f"expected float64, got {batch.dtype}"
        assert batch.ndim == 2
        assert batch.shape == (batch_size, state_len), f"unexpected shape {batch.shape}"


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_as_numpy_amplitudes_are_unit_norm():
    """Each row from as_numpy() should be a unit-norm state vector (amplitude encoding)."""
    num_qubits = 4
    batch_size = 16

    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(num_qubits)
        .batches(2, size=batch_size)
        .source_synthetic()
        .as_numpy()
    )
    for batch in loader:
        arr = np.asarray(batch, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_as_torch_yields_cuda_tensors():
    """as_torch(device='cuda') yields torch tensors on CUDA."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    num_qubits = 4
    batch_size = 8
    state_len = 2**num_qubits

    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(num_qubits)
        .batches(2, size=batch_size)
        .source_synthetic()
        .as_torch(device="cuda")
    )
    for batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.is_cuda
        assert batch.shape == (batch_size, state_len)


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_as_numpy_from_source_array():
    """as_numpy() works with source_array(), yielding correct shapes and dtype."""
    num_qubits = 3
    state_len = 2**num_qubits  # 8
    n_samples = 12
    batch_size = 4

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, state_len))

    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(num_qubits)
        .batches(1, size=batch_size)
        .encoding("amplitude")
        .source_array(X)
        .as_numpy()
    )
    batches = list(loader)
    assert len(batches) == n_samples // batch_size
    for batch in batches:
        assert isinstance(batch, np.ndarray)
        assert batch.dtype == np.float64
        assert batch.shape[1] == state_len
