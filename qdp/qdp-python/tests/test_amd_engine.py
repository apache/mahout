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

import pytest

torch = pytest.importorskip("torch")
_qdp = pytest.importorskip("_qdp")


def _rocm_available() -> bool:
    return bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()


@pytest.mark.skipif(
    not hasattr(_qdp, "AmdQdpEngine"), reason="AmdQdpEngine not available"
)
@pytest.mark.skipif(not _rocm_available(), reason="ROCm runtime not available")
def test_amd_engine_amplitude_shape_and_dtype():
    engine = _qdp.AmdQdpEngine(0, precision="float32")
    x = torch.randn(2, 4, device="cuda", dtype=torch.float32)
    qt = engine.encode(x, 2, "amplitude")
    out = torch.from_dlpack(qt)
    assert out.shape == (2, 4)
    assert out.dtype == torch.complex64


@pytest.mark.skipif(
    not hasattr(_qdp, "AmdQdpEngine"), reason="AmdQdpEngine not available"
)
@pytest.mark.skipif(not _rocm_available(), reason="ROCm runtime not available")
def test_amd_engine_basis_shape_and_values():
    engine = _qdp.AmdQdpEngine(0, precision="float64")
    idx = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    qt = engine.encode(idx, 2, "basis")
    out = torch.from_dlpack(qt)
    assert out.shape == (2, 4)
    assert out.dtype == torch.complex128
    assert out[0, 0].real.item() == pytest.approx(1.0)
    assert out[1, 3].real.item() == pytest.approx(1.0)
