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

from qumat.qumat import QuMat


def test_unsupported_backend_raises_value_error():
    config = {"backend_name": "pennylane", "backend_options": {}}

    with pytest.raises(ValueError) as exc_info:
        QuMat(config)

    error_msg = str(exc_info.value)
    assert "Unsupported backend_name 'pennylane'." in error_msg
    assert "Supported backends: amazon_braket, cirq, qiskit" in error_msg
