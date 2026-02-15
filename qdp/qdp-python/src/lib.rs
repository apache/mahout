//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod dlpack;
mod engine;
mod loader;
mod pytorch;
mod tensor;

use engine::QdpEngine;
use pyo3::prelude::*;
use tensor::QuantumTensor;

#[cfg(target_os = "linux")]
use loader::PyQuantumLoader;

/// Quantum Data Plane (QDP) Python module
///
/// GPU-accelerated quantum data encoding with DLPack integration.
#[pymodule]
fn _qdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Respect RUST_LOG for Rust log output; try_init() is no-op if already initialized.
    let _ = env_logger::Builder::from_default_env().try_init();

    m.add_class::<QdpEngine>()?;
    m.add_class::<QuantumTensor>()?;
    #[cfg(target_os = "linux")]
    m.add_class::<PyQuantumLoader>()?;
    Ok(())
}
