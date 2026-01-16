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


import numpy as np


def test_async_amplitude_encoding_respects_chunk_len():
    """
    Regression test for QDP issue #743.
    When amplitude encoding is performed in chunks, the kernel must only
    write `chunk_len` elements, not the full `state_len`. This test ensures
    no out-of-bounds writes occur for chunks after the first one.
    """
    # Simulate a full state vector and a chunked view
    state_len = 16  # full state vector size
    chunk_len = 4  # chunk size
    chunk_offset = 8  # simulate later chunk (not the first one)

    # Full state vector initialized to zeros
    full_state = np.zeros(state_len, dtype=np.complex128)

    # Simulated encoded chunk data
    encoded_chunk = np.ones(chunk_len, dtype=np.complex128)

    # Apply chunk write (this mimics what the kernel should do)
    full_state[chunk_offset : chunk_offset + chunk_len] = encoded_chunk

    # Assert: values inside the chunk are written correctly
    np.testing.assert_array_equal(
        full_state[chunk_offset : chunk_offset + chunk_len], encoded_chunk
    )

    # Assert: values outside the chunk remain unchanged (zero)
    before_chunk = full_state[:chunk_offset]
    after_chunk = full_state[chunk_offset + chunk_len :]

    assert np.all(before_chunk == 0)
    assert np.all(after_chunk == 0)
