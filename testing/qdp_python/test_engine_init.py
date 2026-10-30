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

"""QdpEngine must raise, not abort the interpreter, when there is no GPU.

Before the guard in ``QdpEngine::new_with_precision`` this test could not
fail politely: constructing the engine on a host without an NVIDIA driver
panicked inside cudarc and the extension's ``panic = "abort"`` profile took
the whole pytest process down. It runs wherever ``_qdp`` imports (conftest
skips it otherwise), including stub builds on GPU-less laptops and CI.
"""

import pytest


def test_engine_without_gpu_raises_runtime_error():
    import _qdp
    from qumat_qdp import is_cuda_available

    if is_cuda_available():
        pytest.skip("a CUDA device is available; nothing to check")
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        _qdp.QdpEngine(0)


def test_engine_without_gpu_float64_raises_runtime_error():
    import _qdp
    from qumat_qdp import is_cuda_available

    if is_cuda_available():
        pytest.skip("a CUDA device is available; nothing to check")
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        _qdp.QdpEngine(0, precision="float64")
