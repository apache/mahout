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

import pytest

try:
    from qumat_qdp.loader import QuantumDataLoader
except ImportError:
    QuantumDataLoader = None  # type: ignore[assignment,misc]


def _loader_available():
    return QuantumDataLoader is not None


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_mutual_exclusion_both_sources_raises():
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
    assert "source_synthetic" in msg and "source_file" in msg


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_mutual_exclusion_exact_message():
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
def test_source_file_empty_path_raises():
    """source_file() with empty path raises ValueError."""
    loader = QuantumDataLoader(device_id=0).qubits(4).batches(10, size=4)
    with pytest.raises(ValueError) as exc_info:
        loader.source_file("")
    assert "path" in str(exc_info.value).lower()


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_synthetic_loader_batch_count():
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
def test_file_loader_unsupported_extension_raises():
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
def test_streaming_requires_parquet():
    """source_file(path, streaming=True) with non-.parquet path raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        QuantumDataLoader(device_id=0).qubits(4).batches(10, size=4).source_file(
            "/tmp/data.npy", streaming=True
        )
    msg = str(exc_info.value).lower()
    assert "parquet" in msg or "streaming" in msg


@pytest.mark.skipif(not _loader_available(), reason="QuantumDataLoader not available")
def test_streaming_parquet_extension_ok():
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
