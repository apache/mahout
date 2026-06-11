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

//! Device runtime surface, vendor-selected at compile time.
//!
//! `cudarc` is CUDA-only with no ROCm backend, so the AMD build cannot depend
//! on it. This module is the seam: on the default (`cuda`) feature it simply
//! re-exports the slice of `cudarc::driver` the crates use; on the `hip`
//! feature it provides a thin HIP-runtime shim with the SAME type names and
//! method signatures, so every call site (`device.alloc`, `htod_sync_copy`,
//! `slice.device_ptr()`, ...) compiles unchanged on both vendors.
//!
//! The marker traits `DeviceRepr` / `ValidAsZeroBits` live here (not in
//! qdp-core) because qdp-kernels implements them on its complex structs and is
//! the lowest crate in the workspace.

#[cfg(not(any(feature = "cuda", feature = "hip")))]
compile_error!("qdp-kernels requires exactly one of the `cuda` or `hip` features");

#[cfg(all(feature = "cuda", not(feature = "hip")))]
pub use cudarc::driver::{
    CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, ValidAsZeroBits,
    safe::CudaStream,
};

#[cfg(feature = "hip")]
pub use hip::{
    CudaDevice, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice,
    DriverError, ValidAsZeroBits,
};

#[cfg(feature = "hip")]
mod hip {
    use std::ffi::c_void;
    use std::marker::PhantomData;
    use std::sync::Arc;

    // ---- HIP runtime FFI (subset used by the device abstraction) ----
    // hip* names map 1:1 to the cuda* runtime entry points cudarc wraps; HIP
    // error codes match CUDA's numerically for the codes we surface.
    #[allow(non_camel_case_types)]
    type hipError_t = i32;

    const HIP_SUCCESS: hipError_t = 0;
    const HIP_MEMCPY_HOST_TO_DEVICE: u32 = 1;
    const HIP_MEMCPY_DEVICE_TO_HOST: u32 = 2;
    // hipStreamNonBlocking: the new stream does not implicitly synchronize with
    // the NULL/default stream, matching cudarc's fork_default_stream.
    // Used on Linux only; Windows uses a blocking stream (see fork_default_stream).
    #[cfg(target_os = "linux")]
    const HIP_STREAM_NON_BLOCKING: u32 = 1;

    unsafe extern "C" {
        fn hipSetDevice(device: i32) -> hipError_t;
        fn hipGetDeviceCount(count: *mut i32) -> hipError_t;
        fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
        fn hipFree(ptr: *mut c_void) -> hipError_t;
        fn hipMemset(ptr: *mut c_void, value: i32, size: usize) -> hipError_t;
        fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: u32) -> hipError_t;
        fn hipDeviceSynchronize() -> hipError_t;
        #[cfg(target_os = "linux")]
        fn hipStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> hipError_t;
        #[cfg(not(target_os = "linux"))]
        fn hipStreamCreate(stream: *mut *mut c_void) -> hipError_t;
        fn hipStreamDestroy(stream: *mut c_void) -> hipError_t;
        fn hipStreamSynchronize(stream: *mut c_void) -> hipError_t;
    }

    /// Mirrors the role of `cudarc::driver::DriverError`: an opaque, `Debug`able
    /// wrapper over a runtime status code. Call sites only ever `{:?}`-format it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DriverError(pub hipError_t);

    fn check(code: hipError_t) -> Result<(), DriverError> {
        if code == HIP_SUCCESS {
            Ok(())
        } else {
            Err(DriverError(code))
        }
    }

    /// Marker: type is safe to byte-copy to/from the device. Mirrors
    /// `cudarc::driver::DeviceRepr`.
    ///
    /// # Safety
    /// Implementor must be `#[repr(C)]`/`#[repr(transparent)]` plain-old-data
    /// with no padding that would expose uninitialized bytes.
    pub unsafe trait DeviceRepr: Copy {}
    unsafe impl DeviceRepr for f32 {}
    unsafe impl DeviceRepr for f64 {}
    unsafe impl DeviceRepr for i32 {}
    unsafe impl DeviceRepr for u32 {}
    unsafe impl DeviceRepr for usize {}

    /// Marker: an all-zero bit pattern is a valid value (enables alloc_zeros).
    /// Mirrors `cudarc::driver::ValidAsZeroBits`.
    ///
    /// # Safety
    /// All-zero bytes must be a valid inhabitant of the type.
    pub unsafe trait ValidAsZeroBits {}
    unsafe impl ValidAsZeroBits for f32 {}
    unsafe impl ValidAsZeroBits for f64 {}
    unsafe impl ValidAsZeroBits for i32 {}
    unsafe impl ValidAsZeroBits for u32 {}
    unsafe impl ValidAsZeroBits for usize {}

    /// Raw device-pointer accessors, matching cudarc's traits. The returned
    /// reference is to the device address stored as `u64`, so the existing
    /// `*slice.device_ptr() as *mut T` call sites work verbatim.
    pub trait DevicePtr<T> {
        fn device_ptr(&self) -> &u64;
    }
    pub trait DevicePtrMut<T> {
        fn device_ptr_mut(&mut self) -> &mut u64;
    }
    /// Length accessor, matching cudarc's `DeviceSlice`.
    pub trait DeviceSlice<T> {
        fn len(&self) -> usize;
        fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }

    /// Owned device allocation; frees on drop. Stand-in for `cudarc::CudaSlice`.
    pub struct CudaSlice<T> {
        ptr: u64,
        len: usize,
        _device: Arc<CudaDevice>,
        _marker: PhantomData<T>,
    }

    // The device address is just an integer; ownership/lifetime is enforced by
    // the held Arc<CudaDevice>. Safe to move across threads like cudarc's slice.
    unsafe impl<T: Send> Send for CudaSlice<T> {}
    unsafe impl<T: Sync> Sync for CudaSlice<T> {}

    impl<T> CudaSlice<T> {
        fn raw_ptr(&self) -> *mut c_void {
            self.ptr as *mut c_void
        }

        /// Mutable sub-view `[range.start, range.end)`. Mirrors
        /// `cudarc::CudaSlice::slice_mut`; the returned view borrows this slice
        /// and is itself a `DevicePtrMut`/`DeviceSlice` copy target.
        pub fn slice_mut(&mut self, range: std::ops::Range<usize>) -> CudaViewMut<'_, T> {
            assert!(
                range.start <= range.end && range.end <= self.len,
                "slice_mut out of bounds"
            );
            let offset_ptr = self.ptr + (range.start * std::mem::size_of::<T>()) as u64;
            CudaViewMut {
                ptr: offset_ptr,
                len: range.end - range.start,
                _parent: PhantomData,
            }
        }
    }

    /// Borrowed mutable view into a `CudaSlice`, returned by `slice_mut`.
    pub struct CudaViewMut<'a, T> {
        ptr: u64,
        len: usize,
        _parent: PhantomData<&'a mut T>,
    }

    impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
        fn device_ptr(&self) -> &u64 {
            &self.ptr
        }
    }
    impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
        fn device_ptr_mut(&mut self) -> &mut u64 {
            &mut self.ptr
        }
    }
    impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
        fn len(&self) -> usize {
            self.len
        }
    }

    impl<T> DevicePtr<T> for CudaSlice<T> {
        fn device_ptr(&self) -> &u64 {
            &self.ptr
        }
    }
    impl<T> DevicePtrMut<T> for CudaSlice<T> {
        fn device_ptr_mut(&mut self) -> &mut u64 {
            &mut self.ptr
        }
    }
    impl<T> DeviceSlice<T> for CudaSlice<T> {
        fn len(&self) -> usize {
            self.len
        }
    }

    impl<T> Drop for CudaSlice<T> {
        fn drop(&mut self) {
            if self.ptr != 0 {
                // hipFree releases on the calling thread's current device, so
                // re-bind the owning device first (cudarc does the same in Drop):
                // on multi-GPU a different device may be current, which would
                // otherwise free against the wrong device. Best-effort -- Drop
                // cannot report an error, so a failed bind is swallowed.
                let _ = self._device.bind();
                unsafe {
                    let _ = hipFree(self.raw_ptr());
                }
            }
        }
    }

    /// A HIP stream. The public `stream` field mirrors cudarc's
    /// `CudaStream { stream: sys::CUstream, .. }` so existing call sites that do
    /// `ctx.stream_compute.stream as *mut c_void` keep working.
    pub struct CudaStream {
        pub stream: *mut c_void,
        _device: Arc<CudaDevice>,
    }

    unsafe impl Send for CudaStream {}
    unsafe impl Sync for CudaStream {}

    impl Drop for CudaStream {
        fn drop(&mut self) {
            if !self.stream.is_null() {
                unsafe {
                    let _ = hipStreamDestroy(self.stream);
                }
            }
        }
    }

    /// HIP device handle. Stand-in for `cudarc::CudaDevice`; created via
    /// `CudaDevice::new(ordinal)` and shared as `Arc<CudaDevice>` exactly like
    /// cudarc (whose `new` already returns the `Arc`).
    pub struct CudaDevice {
        ordinal: usize,
    }

    impl CudaDevice {
        /// Select device `ordinal` and return a shared handle, or an error if no
        /// such device exists. Matches `cudarc::CudaDevice::new`.
        pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
            unsafe {
                let mut count: i32 = 0;
                check(hipGetDeviceCount(&mut count))?;
                if ordinal as i32 >= count {
                    return Err(DriverError(101)); // hipErrorInvalidDevice
                }
                check(hipSetDevice(ordinal as i32))?;
            }
            Ok(Arc::new(Self { ordinal }))
        }

        pub fn ordinal(&self) -> usize {
            self.ordinal
        }

        fn bind(&self) -> Result<(), DriverError> {
            unsafe { check(hipSetDevice(self.ordinal as i32)) }
        }

        /// Allocate `len` uninitialized elements of `T` on the device.
        ///
        /// # Safety
        /// Contents are uninitialized until written, mirroring `cudarc`'s
        /// `unsafe fn alloc`.
        pub unsafe fn alloc<T>(self: &Arc<Self>, len: usize) -> Result<CudaSlice<T>, DriverError> {
            self.bind()?;
            let bytes = len.saturating_mul(std::mem::size_of::<T>());
            let mut ptr: *mut c_void = std::ptr::null_mut();
            unsafe {
                check(hipMalloc(&mut ptr, bytes.max(1)))?;
            }
            Ok(CudaSlice {
                ptr: ptr as u64,
                len,
                _device: Arc::clone(self),
                _marker: PhantomData,
            })
        }

        /// Allocate `len` zero-initialized elements of `T` on the device.
        pub fn alloc_zeros<T: ValidAsZeroBits>(
            self: &Arc<Self>,
            len: usize,
        ) -> Result<CudaSlice<T>, DriverError> {
            let slice = unsafe { self.alloc::<T>(len)? };
            let bytes = len.saturating_mul(std::mem::size_of::<T>());
            if bytes > 0 {
                unsafe {
                    check(hipMemset(slice.raw_ptr(), 0, bytes))?;
                }
            }
            Ok(slice)
        }

        /// Copy a host slice to a freshly allocated device buffer (blocking).
        pub fn htod_sync_copy<T: DeviceRepr>(
            self: &Arc<Self>,
            src: &[T],
        ) -> Result<CudaSlice<T>, DriverError> {
            let mut slice = unsafe { self.alloc::<T>(src.len())? };
            self.htod_sync_copy_into(src, &mut slice)?;
            Ok(slice)
        }

        /// Copy an owned host Vec to a freshly allocated device buffer. cudarc's
        /// `htod_copy` keeps the Vec alive until an async copy completes; our
        /// copy is synchronous (blocking hipMemcpy), so the Vec can be dropped
        /// on return with identical observable behavior.
        pub fn htod_copy<T: DeviceRepr>(
            self: &Arc<Self>,
            src: Vec<T>,
        ) -> Result<CudaSlice<T>, DriverError> {
            self.htod_sync_copy(&src)
        }

        /// Copy a host slice into an existing device buffer or sub-view
        /// (blocking). Accepts any `DevicePtrMut` target so both `CudaSlice` and
        /// the `slice_mut` view work, matching cudarc's generic destination.
        pub fn htod_sync_copy_into<T: DeviceRepr, D: DevicePtrMut<T> + DeviceSlice<T>>(
            self: &Arc<Self>,
            src: &[T],
            dst: &mut D,
        ) -> Result<(), DriverError> {
            assert_eq!(
                dst.len(),
                src.len(),
                "htod_sync_copy_into: dst.len() != src.len()"
            );
            self.bind()?;
            let bytes = std::mem::size_of_val(src);
            if bytes > 0 {
                unsafe {
                    check(hipMemcpy(
                        (*dst.device_ptr_mut()) as *mut c_void,
                        src.as_ptr() as *const c_void,
                        bytes,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    ))?;
                }
            }
            Ok(())
        }

        /// Copy a device buffer back to a freshly allocated host Vec (blocking).
        ///
        /// Matches cudarc's bound of just `DeviceRepr` (no `Default`): the Vec is
        /// allocated uninitialized and every byte is written by the copy before
        /// its length is set, which is sound because `DeviceRepr` is plain data.
        pub fn dtoh_sync_copy<T: DeviceRepr>(
            self: &Arc<Self>,
            src: &CudaSlice<T>,
        ) -> Result<Vec<T>, DriverError> {
            self.bind()?;
            let len = src.len;
            let mut out: Vec<T> = Vec::with_capacity(len);
            let bytes = len.saturating_mul(std::mem::size_of::<T>());
            if bytes > 0 {
                unsafe {
                    check(hipMemcpy(
                        out.as_mut_ptr() as *mut c_void,
                        src.raw_ptr() as *const c_void,
                        bytes,
                        HIP_MEMCPY_DEVICE_TO_HOST,
                    ))?;
                }
            }
            unsafe {
                out.set_len(len);
            }
            Ok(out)
        }

        /// Block until all work on the device's default stream completes.
        pub fn synchronize(&self) -> Result<(), DriverError> {
            self.bind()?;
            unsafe { check(hipDeviceSynchronize()) }
        }

        /// Create a new stream tied to this device. Mirrors
        /// `cudarc::CudaDevice::fork_default_stream`.
        pub fn fork_default_stream(self: &Arc<Self>) -> Result<CudaStream, DriverError> {
            self.bind()?;
            let mut stream: *mut c_void = std::ptr::null_mut();
            unsafe {
                // On Linux, use a non-blocking stream (HIP_STREAM_NON_BLOCKING=1) to
                // match cudarc: a blocking stream serializes H2D copies against the NULL
                // stream and defeats the dual-stream copy/compute overlap the pipeline
                // relies on.
                //
                // On Windows (TheRock ROCm 7.14), non-blocking stream writes are not
                // visible via hipMemcpy from the host after hipStreamSynchronize, due to
                // a cache coherency gap in the Windows HIP runtime. Use a blocking stream
                // to restore correct D2H readback semantics. The pipeline overlap
                // optimization is not available on Windows ROCm.
                #[cfg(target_os = "linux")]
                check(hipStreamCreateWithFlags(
                    &mut stream,
                    HIP_STREAM_NON_BLOCKING,
                ))?;
                #[cfg(not(target_os = "linux"))]
                check(hipStreamCreate(&mut stream))?;
            }
            Ok(CudaStream {
                stream,
                _device: Arc::clone(self),
            })
        }

        /// Block until all work on `stream` completes. Mirrors
        /// `cudarc::CudaDevice::wait_for`.
        pub fn wait_for(&self, stream: &CudaStream) -> Result<(), DriverError> {
            unsafe { check(hipStreamSynchronize(stream.stream)) }
        }
    }
}
