
<!--
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Apache Mahout Testing Suite

For each backend supported in Apache Mahout, the testing suite executes an example circuit using the qumat implementation of the backend, and then executes the same example circuit using the backend's native implementation. The test then checks that the resulting final state vectors are the same.

The testing suite is run using `pytest`, which is installed by default. To run the tests, simply run

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v
```

## Adding New Backend Tests

1. Create `testing/my-new-backend_helpers.py` with:
   - `get_qumat_backend_config()` function which returns the qumat backend config needed for the qumat implementation of my-new-backend
   - `get_native_example_final_state_vector()` function which builds and executes the example circuit using the native implementation of my-new-backend

2. Add `"my-new-backend"` to `testing_backends` fixture in `conftest.py`
