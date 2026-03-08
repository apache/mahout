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

use pyo3::prelude::*;

/// Helper to detect Jax array
pub fn is_jax_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let type_obj = obj.get_type();
    
    // Jax arrays usually belong to the 'jaxlib' or 'jax' modules.
    let module = type_obj.module()?;
    let module_name = module.to_str()?;
    
    Ok(module_name.starts_with("jaxlib") || module_name.starts_with("jax"))
}

/// Helper to synchronize Jax array before DLPack extraction.
/// equivalent to `jax_array.block_until_ready()`
pub fn synchronize_jax_array(jax_array: &Bound<'_, PyAny>) -> PyResult<()> {
    if jax_array.hasattr("block_until_ready")? {
        jax_array.call_method0("block_until_ready")?;
    }
    Ok(())
}
