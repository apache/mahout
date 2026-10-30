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

//! Every embedded kernel must resolve, and the generic launcher must pass
//! arguments with the right ABI. These are the checks that replace matching
//! `extern "C"` declarations by hand.

#![cfg(target_os = "linux")]

use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
use qdp_kernels::{CuComplex, LaunchConfig, kernel_args};

fn device() -> Option<std::sync::Arc<CudaDevice>> {
    if !qdp_kernels::kernels_embedded() {
        println!("SKIP: built without CUDA toolkit");
        return None;
    }
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(_) => {
            println!("SKIP: no CUDA device available");
            None
        }
    }
}

#[test]
fn every_embedded_symbol_resolves() {
    let Some(device) = device() else { return };
    let mut count = 0;
    for module in qdp_kernels::module_names() {
        let symbols = qdp_kernels::module_symbols(module).unwrap();
        assert!(!symbols.is_empty(), "module {module} declares no kernels");
        for name in symbols {
            qdp_kernels::function(&device, module, name)
                .unwrap_or_else(|e| panic!("{module}::{name} failed to resolve: {e}"));
            count += 1;
        }
    }
    assert!(
        count >= 30,
        "expected the full kernel set, resolved {count}"
    );
}

#[test]
fn unknown_symbol_is_reported_before_touching_the_driver() {
    let Some(device) = device() else { return };
    let err = qdp_kernels::function(&device, "angle", "no_such_kernel").unwrap_err();
    assert!(
        matches!(err, qdp_kernels::KernelError::UnknownSymbol { .. }),
        "{err}"
    );
    let err = qdp_kernels::function(&device, "no_such_module", "x").unwrap_err();
    assert!(
        matches!(err, qdp_kernels::KernelError::UnknownModule(_)),
        "{err}"
    );
}

#[test]
fn generic_launch_passes_arguments_in_order() {
    let Some(device) = device() else { return };

    // |psi> = (cos π/4 |0> + sin π/4 |1>) ⊗ (|0>)  =>  [c, c, 0, 0] with c = 1/√2
    let angles = device
        .htod_copy(vec![std::f32::consts::FRAC_PI_4, 0.0_f32])
        .unwrap();
    let mut state = device.alloc_zeros::<CuComplex>(4).unwrap();
    let state_len: usize = 4;
    let num_qubits: u32 = 2;

    let f = qdp_kernels::function(&device, "angle", "angle_encode_kernel_f32").unwrap();
    unsafe {
        qdp_kernels::launch(
            &device,
            f,
            LaunchConfig::grid_1d(state_len),
            std::ptr::null_mut(),
            &mut kernel_args![
                *angles.device_ptr() as *const f32,
                *state.device_ptr_mut() as *mut CuComplex,
                state_len,
                num_qubits
            ],
        )
        .unwrap();
    }
    device.synchronize().unwrap();

    let host = device.dtoh_sync_copy(&state).unwrap();
    let c = std::f32::consts::FRAC_1_SQRT_2;
    let expected = [c, c, 0.0, 0.0];
    for (i, (got, want)) in host.iter().zip(expected).enumerate() {
        assert!(
            (got.x - want).abs() < 1e-6,
            "state[{i}].x = {}, want {want}",
            got.x
        );
        assert!(got.y.abs() < 1e-6, "state[{i}].y = {}", got.y);
    }
}
