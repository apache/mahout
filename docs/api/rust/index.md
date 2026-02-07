---
title: Rust API Reference
sidebar_label: Rust API (qdp-core)
sidebar_position: 2
---

# Crate Documentation

**Version:** 0.2.0

**Format Version:** 57

# Module `qdp_core`

## Modules

## Module `dlpack`

```rust
pub mod dlpack { /* ... */ }
```

### Types

#### Enum `DLDeviceType`

**Attributes:**

- `Other("#[allow(non_camel_case_types)]")`
- `Repr(AttributeRepr { kind: C, align: None, packed: None, int: None })`

Device type enum for DLPack. Eq/PartialEq used for validation (e.g. device_type != kDLCUDA);
Debug for diagnostics; Copy/Clone for FFI ergonomics when used in DLDevice.

```rust
pub enum DLDeviceType {
    kDLCPU = 1,
    kDLCUDA = 2,
}
```

##### Variants

###### `kDLCPU`

Discriminant: `1`

Discriminant value: `1`

###### `kDLCUDA`

Discriminant: `2`

Discriminant value: `2`

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Clone**
  - ```rust
    fn clone(self: &Self) -> DLDeviceType { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Copy**
- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut $crate::fmt::Formatter<''_>) -> $crate::fmt::Result { /* ... */ }
    ```

- **Eq**
- **Equivalent**
  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &DLDeviceType) -> bool { /* ... */ }
    ```

- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Struct `DLDevice`

**Attributes:**

- `Repr(AttributeRepr { kind: C, align: None, packed: None, int: None })`

```rust
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: std::os::raw::c_int,
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `device_type` | `DLDeviceType` |  |
| `device_id` | `std::os::raw::c_int` |  |

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Struct `DLDataType`

**Attributes:**

- `Repr(AttributeRepr { kind: C, align: None, packed: None, int: None })`

```rust
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `code` | `u8` |  |
| `bits` | `u8` |  |
| `lanes` | `u16` |  |

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Struct `DLTensor`

**Attributes:**

- `Repr(AttributeRepr { kind: C, align: None, packed: None, int: None })`

```rust
pub struct DLTensor {
    pub data: *mut std::os::raw::c_void,
    pub device: DLDevice,
    pub ndim: std::os::raw::c_int,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `data` | `*mut std::os::raw::c_void` |  |
| `device` | `DLDevice` |  |
| `ndim` | `std::os::raw::c_int` |  |
| `dtype` | `DLDataType` |  |
| `shape` | `*mut i64` |  |
| `strides` | `*mut i64` |  |
| `byte_offset` | `u64` |  |

##### Implementations

###### Trait Implementations

- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Struct `DLManagedTensor`

**Attributes:**

- `Repr(AttributeRepr { kind: C, align: None, packed: None, int: None })`

```rust
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut std::os::raw::c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `dl_tensor` | `DLTensor` |  |
| `manager_ctx` | `*mut std::os::raw::c_void` |  |
| `deleter` | `Option<unsafe extern "C" fn(*mut DLManagedTensor)>` |  |

##### Implementations

###### Trait Implementations

- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
### Functions

#### Function `dlpack_stream_to_cuda`

Map DLPack stream integer to a CUDA stream pointer.

```rust
pub fn dlpack_stream_to_cuda(stream: i64) -> *mut std::os::raw::c_void { /* ... */ }
```

#### Function `synchronize_stream`

**Attributes:**

- `Other("#[attr = CfgTrace([Not(NameValue { name: \"target_os\", value: Some(\"linux\"), span: qdp-core/src/dlpack.rs:99:11: 99:30 (#0) }, qdp-core/src/dlpack.rs:99:10: 99:31 (#0))])]")`

# Safety
No-op on non-Linux targets, kept unsafe to match the Linux signature.

```rust
pub unsafe fn synchronize_stream(_stream: *mut std::os::raw::c_void) -> crate::error::Result<()> { /* ... */ }
```

#### Function `dlpack_deleter`

**Attributes:**

- `Other("#[allow(unsafe_op_in_unsafe_fn)]")`

Called by PyTorch to free tensor memory

# Safety
Frees shape, strides, GPU buffer, and managed tensor.
Caller must ensure the pointer is valid and points to a properly initialized DLManagedTensor.

```rust
pub unsafe extern "C" fn dlpack_deleter(managed: *mut DLManagedTensor) { /* ... */ }
```

### Constants and Statics

#### Constant `CUDA_STREAM_LEGACY`

**Attributes:**

- `Other("#[allow(clippy::manual_dangling_ptr)]")`

DLPack CUDA stream sentinel values (legacy/per-thread default).
These match cudaStreamLegacy/cudaStreamPerThread in the CUDA runtime.

```rust
pub const CUDA_STREAM_LEGACY: *mut std::os::raw::c_void = _;
```

#### Constant `CUDA_STREAM_PER_THREAD`

**Attributes:**

- `Other("#[allow(clippy::manual_dangling_ptr)]")`

```rust
pub const CUDA_STREAM_PER_THREAD: *mut std::os::raw::c_void = _;
```

#### Constant `DL_INT`

**Attributes:**

- `Other("#[allow(dead_code)]")`

```rust
pub const DL_INT: u8 = 0;
```

#### Constant `DL_UINT`

**Attributes:**

- `Other("#[allow(dead_code)]")`

```rust
pub const DL_UINT: u8 = 1;
```

#### Constant `DL_FLOAT`

**Attributes:**

- `Other("#[allow(dead_code)]")`

```rust
pub const DL_FLOAT: u8 = 2;
```

#### Constant `DL_BFLOAT`

**Attributes:**

- `Other("#[allow(dead_code)]")`

```rust
pub const DL_BFLOAT: u8 = 4;
```

#### Constant `DL_COMPLEX`

```rust
pub const DL_COMPLEX: u8 = 5;
```

## Module `error`

```rust
pub mod error { /* ... */ }
```

### Types

#### Enum `MahoutError`

Error types for Mahout QDP operations

```rust
pub enum MahoutError {
    Cuda(String),
    InvalidInput(String),
    MemoryAllocation(String),
    KernelLaunch(String),
    DLPack(String),
    Io(String),
    NotImplemented(String),
}
```

##### Variants

###### `Cuda`

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `String` |  |

###### `InvalidInput`

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `String` |  |

###### `MemoryAllocation`

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `String` |  |

###### `KernelLaunch`

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `String` |  |

###### `DLPack`

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `String` |  |

###### `Io`

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `String` |  |

###### `NotImplemented`

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `String` |  |

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut $crate::fmt::Formatter<''_>) -> $crate::fmt::Result { /* ... */ }
    ```

- **Display**
  - ```rust
    fn fmt(self: &Self, __formatter: &mut ::core::fmt::Formatter<''_>) -> ::core::fmt::Result { /* ... */ }
    ```

- **Error**
- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **ToString**
  - ```rust
    fn to_string(self: &Self) -> String { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Type Alias `Result`

Result type alias for Mahout operations

```rust
pub type Result<T> = std::result::Result<T, MahoutError>;
```

## Module `gpu`

```rust
pub mod gpu { /* ... */ }
```

### Modules

## Module `encodings`

```rust
pub mod encodings { /* ... */ }
```

### Modules

## Module `amplitude`

**Attributes:**

- `Other("#[allow(unused_unsafe)]")`

```rust
pub mod amplitude { /* ... */ }
```

### Types

#### Struct `AmplitudeEncoder`

Amplitude encoding: data → normalized quantum amplitudes

Steps: L2 norm (CPU) → GPU allocation → CUDA kernel (normalize + pad)
Fast: ~50-100x vs circuit-based methods

```rust
pub struct AmplitudeEncoder;
```

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **QuantumEncoder**
  - ```rust
    fn encode(self: &Self, _device: &Arc<CudaDevice>, host_data: &[f64], num_qubits: usize) -> Result<GpuStateVector> { /* ... */ }
    ```

  - ```rust
    fn name(self: &Self) -> &''static str { /* ... */ }
    ```

  - ```rust
    fn description(self: &Self) -> &''static str { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Module `angle`

**Attributes:**

- `Other("#[allow(unused_unsafe)]")`

```rust
pub mod angle { /* ... */ }
```

### Types

#### Struct `AngleEncoder`

Angle encoding: each qubit uses one rotation angle to form a product state.

```rust
pub struct AngleEncoder;
```

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **QuantumEncoder**
  - ```rust
    fn encode(self: &Self, _device: &Arc<CudaDevice>, data: &[f64], num_qubits: usize) -> Result<GpuStateVector> { /* ... */ }
    ```

  - ```rust
    fn validate_input(self: &Self, data: &[f64], num_qubits: usize) -> Result<()> { /* ... */ }
    ```

  - ```rust
    fn name(self: &Self) -> &''static str { /* ... */ }
    ```

  - ```rust
    fn description(self: &Self) -> &''static str { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Module `basis`

**Attributes:**

- `Other("#[allow(unused_unsafe)]")`

```rust
pub mod basis { /* ... */ }
```

### Types

#### Struct `BasisEncoder`

Basis encoding: maps an integer index to a computational basis state.

For n qubits, maps integer i (0 ≤ i < 2^n) to |i⟩, where:
- state[i] = 1.0 + 0.0i
- state[j] = 0.0 + 0.0i for all j ≠ i

Example: index 3 with 3 qubits → |011⟩ (binary representation of 3)

Input format:
- Single encoding: data = [index] (single f64 representing the basis index)
- Batch encoding: data = [idx0, idx1, ..., idxN] (one index per sample)

```rust
pub struct BasisEncoder;
```

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **QuantumEncoder**
  - ```rust
    fn encode(self: &Self, _device: &Arc<CudaDevice>, data: &[f64], num_qubits: usize) -> Result<GpuStateVector> { /* ... */ }
    ```

  - ```rust
    fn validate_input(self: &Self, data: &[f64], num_qubits: usize) -> Result<()> { /* ... */ }
    ```

  - ```rust
    fn name(self: &Self) -> &''static str { /* ... */ }
    ```

  - ```rust
    fn description(self: &Self) -> &''static str { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Module `iqp`

```rust
pub mod iqp { /* ... */ }
```

### Types

#### Struct `IqpEncoder`

IQP encoding: creates entangled quantum states using diagonal phase gates.

Two variants are supported:
- `enable_zz = false`: Single-qubit Z rotations only (n parameters)
- `enable_zz = true`: Full ZZ interactions (n + n*(n-1)/2 parameters)

```rust
pub struct IqpEncoder {
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn full() -> Self { /* ... */ }
  ```
  Create an IQP encoder with full ZZ interactions.

- ```rust
  pub fn z_only() -> Self { /* ... */ }
  ```
  Create an IQP encoder with single-qubit Z rotations only.

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **QuantumEncoder**
  - ```rust
    fn encode(self: &Self, _device: &Arc<CudaDevice>, data: &[f64], num_qubits: usize) -> Result<GpuStateVector> { /* ... */ }
    ```

  - ```rust
    fn validate_input(self: &Self, data: &[f64], num_qubits: usize) -> Result<()> { /* ... */ }
    ```

  - ```rust
    fn name(self: &Self) -> &''static str { /* ... */ }
    ```

  - ```rust
    fn description(self: &Self) -> &''static str { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
### Traits

#### Trait `QuantumEncoder`

Quantum encoding strategy interface
Implemented by: AmplitudeEncoder, AngleEncoder, BasisEncoder

```rust
pub trait QuantumEncoder: Send + Sync {
    /* Associated items */
}
```

##### Required Items

###### Required Methods

- `encode`: Encode classical data to quantum state on GPU
- `name`: Strategy name
- `description`: Strategy description

##### Provided Methods

- ```rust
  fn encode_batch(self: &Self, _device: &Arc<CudaDevice>, _batch_data: &[f64], _num_samples: usize, _sample_size: usize, _num_qubits: usize) -> Result<GpuStateVector> { /* ... */ }
  ```
  Encode multiple samples in a single GPU allocation and kernel launch (Batch Encoding)

- ```rust
  fn validate_input(self: &Self, data: &[f64], num_qubits: usize) -> Result<()> { /* ... */ }
  ```
  Validate input data before encoding

##### Implementations

This trait is implemented for the following types:

- `AmplitudeEncoder`
- `AngleEncoder`
- `BasisEncoder`
- `IqpEncoder`

### Functions

#### Function `validate_qubit_count`

Validates qubit count against practical limits.

Checks:
- Qubit count is at least 1
- Qubit count does not exceed MAX_QUBITS

# Arguments
* `num_qubits` - The number of qubits to validate

# Returns
* `Ok(())` if the qubit count is valid
* `Err(MahoutError::InvalidInput)` if the qubit count is invalid

```rust
pub fn validate_qubit_count(num_qubits: usize) -> crate::error::Result<()> { /* ... */ }
```

#### Function `get_encoder`

Create encoder by name: "amplitude", "angle", "basis", "iqp", or "iqp-z"

```rust
pub fn get_encoder(name: &str) -> crate::error::Result<Box<dyn QuantumEncoder>> { /* ... */ }
```

### Constants and Statics

#### Constant `MAX_QUBITS`

Maximum number of qubits supported (16GB GPU memory limit)
This constant must match MAX_QUBITS in qdp-kernels/src/kernel_config.h

```rust
pub const MAX_QUBITS: usize = 30;
```

### Re-exports

#### Re-export `AmplitudeEncoder`

```rust
pub use amplitude::AmplitudeEncoder;
```

#### Re-export `AngleEncoder`

```rust
pub use angle::AngleEncoder;
```

#### Re-export `BasisEncoder`

```rust
pub use basis::BasisEncoder;
```

#### Re-export `IqpEncoder`

```rust
pub use iqp::IqpEncoder;
```

## Module `memory`

**Attributes:**

- `Other("#[allow(unused_unsafe)]")`

```rust
pub mod memory { /* ... */ }
```

### Types

#### Enum `Precision`

Precision of the GPU state vector.

```rust
pub enum Precision {
    Float32,
    Float64,
}
```

##### Variants

###### `Float32`

###### `Float64`

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Clone**
  - ```rust
    fn clone(self: &Self) -> Precision { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Copy**
- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut $crate::fmt::Formatter<''_>) -> $crate::fmt::Result { /* ... */ }
    ```

- **Eq**
- **Equivalent**
  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &Precision) -> bool { /* ... */ }
    ```

- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Struct `GpuBufferRaw`

RAII wrapper for GPU memory buffer
Automatically frees GPU memory when dropped

```rust
pub struct GpuBufferRaw<T> {
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn ptr(self: &Self) -> *mut T { /* ... */ }
  ```
  Get raw pointer to GPU memory

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Enum `BufferStorage`

Storage wrapper for precision-specific GPU buffers

```rust
pub enum BufferStorage {
    F32(GpuBufferRaw<qdp_kernels::CuComplex>),
    F64(GpuBufferRaw<qdp_kernels::CuDoubleComplex>),
}
```

##### Variants

###### `F32`

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `GpuBufferRaw<qdp_kernels::CuComplex>` |  |

###### `F64`

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `GpuBufferRaw<qdp_kernels::CuDoubleComplex>` |  |

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Struct `GpuStateVector`

Quantum state vector on GPU

Manages complex array of size 2^n (n = qubits) in GPU memory.
Uses Arc for shared ownership (needed for DLPack/PyTorch integration).
Thread-safe: Send + Sync

```rust
pub struct GpuStateVector {
    pub num_qubits: usize,
    pub size_elements: usize,
    pub device_id: usize,
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `num_qubits` | `usize` |  |
| `size_elements` | `usize` |  |
| `device_id` | `usize` | CUDA device ordinal |
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn to_dlpack(self: &Self) -> *mut DLManagedTensor { /* ... */ }
  ```
  Convert to DLPack format for PyTorch

- ```rust
  pub fn new(_device: &Arc<CudaDevice>, _qubits: usize, _precision: Precision) -> Result<Self> { /* ... */ }
  ```

- ```rust
  pub fn precision(self: &Self) -> Precision { /* ... */ }
  ```
  Get current precision of the underlying buffer.

- ```rust
  pub fn ptr_void(self: &Self) -> *mut c_void { /* ... */ }
  ```
  Get raw GPU pointer for DLPack/FFI

- ```rust
  pub fn ptr_f64(self: &Self) -> Option<*mut CuDoubleComplex> { /* ... */ }
  ```
  Returns a double-precision pointer if the buffer stores complex128 data.

- ```rust
  pub fn ptr_f32(self: &Self) -> Option<*mut CuComplex> { /* ... */ }
  ```
  Returns a single-precision pointer if the buffer stores complex64 data.

- ```rust
  pub fn num_qubits(self: &Self) -> usize { /* ... */ }
  ```
  Get the number of qubits

- ```rust
  pub fn size_elements(self: &Self) -> usize { /* ... */ }
  ```
  Get the size in elements (2^n where n is number of qubits)

- ```rust
  pub fn new_batch(_device: &Arc<CudaDevice>, num_samples: usize, qubits: usize) -> Result<Self> { /* ... */ }
  ```
  Create GPU state vector for a batch of samples

- ```rust
  pub fn to_precision(self: &Self, device: &Arc<CudaDevice>, target: Precision) -> Result<Self> { /* ... */ }
  ```
  Convert the state vector to the requested precision (GPU-side).

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Clone**
  - ```rust
    fn clone(self: &Self) -> GpuStateVector { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Module `pipeline`

**Attributes:**

- `Other("#[allow(unused_unsafe)]")`

```rust
pub mod pipeline { /* ... */ }
```

### Re-exports

#### Re-export `AmplitudeEncoder`

```rust
pub use encodings::AmplitudeEncoder;
```

#### Re-export `AngleEncoder`

```rust
pub use encodings::AngleEncoder;
```

#### Re-export `BasisEncoder`

```rust
pub use encodings::BasisEncoder;
```

#### Re-export `QuantumEncoder`

```rust
pub use encodings::QuantumEncoder;
```

#### Re-export `get_encoder`

```rust
pub use encodings::get_encoder;
```

#### Re-export `GpuStateVector`

```rust
pub use memory::GpuStateVector;
```

## Module `io`

I/O utilities for reading and writing quantum data.

Provides efficient columnar data exchange via Apache Arrow and Parquet formats.

# TODO
Consider using generic `T: ArrowPrimitiveType` instead of hardcoded `Float64Array`
to support both Float32 and Float64 for flexibility in precision vs performance trade-offs.

```rust
pub mod io { /* ... */ }
```

### Types

#### Type Alias `ParquetBlockReader`

Streaming Parquet reader for List<Float64> and FixedSizeList<Float64> columns

Reads Parquet files in chunks without loading entire file into memory.
Supports efficient streaming for large files via Producer-Consumer pattern.

This is a type alias for backward compatibility. Use [`crate::readers::ParquetStreamingReader`] directly.

```rust
pub type ParquetBlockReader = crate::readers::ParquetStreamingReader;
```

### Functions

#### Function `arrow_to_vec`

Converts an Arrow Float64Array to Vec<f64>.

```rust
pub fn arrow_to_vec(array: &arrow::array::Float64Array) -> Vec<f64> { /* ... */ }
```

#### Function `arrow_to_vec_chunked`

Flattens multiple Arrow Float64Arrays into a single Vec<f64>.

```rust
pub fn arrow_to_vec_chunked(arrays: &[arrow::array::Float64Array]) -> Vec<f64> { /* ... */ }
```

#### Function `read_parquet`

Reads Float64 data from a Parquet file.

Expects a single Float64 column. For zero-copy access, use [`read_parquet_to_arrow`].

```rust
pub fn read_parquet<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<Vec<f64>> { /* ... */ }
```

#### Function `write_parquet`

Writes Float64 data to a Parquet file.

# Arguments
* `path` - Output file path
* `data` - Data to write
* `column_name` - Column name (defaults to "data")

```rust
pub fn write_parquet<P: AsRef<std::path::Path>>(path: P, data: &[f64], column_name: Option<&str>) -> crate::error::Result<()> { /* ... */ }
```

#### Function `read_parquet_to_arrow`

Reads a Parquet file as Arrow Float64Arrays.

Returns one array per row group for zero-copy access.

```rust
pub fn read_parquet_to_arrow<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<Vec<arrow::array::Float64Array>> { /* ... */ }
```

#### Function `write_arrow_to_parquet`

Writes an Arrow Float64Array to a Parquet file.

# Arguments
* `path` - Output file path
* `array` - Array to write
* `column_name` - Column name (defaults to "data")

```rust
pub fn write_arrow_to_parquet<P: AsRef<std::path::Path>>(path: P, array: &arrow::array::Float64Array, column_name: Option<&str>) -> crate::error::Result<()> { /* ... */ }
```

#### Function `read_parquet_batch`

Reads batch data from a Parquet file with `List<Float64>` column format.

Returns flattened data suitable for batch encoding.

# Returns
Tuple of `(flattened_data, num_samples, sample_size)`

# TODO
Add OOM protection for very large files

```rust
pub fn read_parquet_batch<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<(Vec<f64>, usize, usize)> { /* ... */ }
```

#### Function `read_arrow_ipc_batch`

Reads batch data from an Arrow IPC file.

Supports `FixedSizeList<Float64>` and `List<Float64>` column formats.
Returns flattened data suitable for batch encoding.

# Returns
Tuple of `(flattened_data, num_samples, sample_size)`

# TODO
Add OOM protection for very large files

```rust
pub fn read_arrow_ipc_batch<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<(Vec<f64>, usize, usize)> { /* ... */ }
```

#### Function `read_numpy_batch`

Reads batch data from a NumPy .npy file.

Expects a 2D array with shape `[num_samples, sample_size]` and dtype `float64`.
Returns flattened data suitable for batch encoding.

# Returns
Tuple of `(flattened_data, num_samples, sample_size)`

# Example
```rust,ignore
let (data, num_samples, sample_size) = read_numpy_batch("quantum_states.npy")?;
```

```rust
pub fn read_numpy_batch<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<(Vec<f64>, usize, usize)> { /* ... */ }
```

#### Function `read_torch_batch`

Reads batch data from a PyTorch .pt/.pth file.

Expects a 1D or 2D tensor saved with `torch.save`.
Returns flattened data suitable for batch encoding.
Requires the `pytorch` feature to be enabled.

# Returns
Tuple of `(flattened_data, num_samples, sample_size)`

```rust
pub fn read_torch_batch<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<(Vec<f64>, usize, usize)> { /* ... */ }
```

#### Function `read_tensorflow_batch`

Reads batch data from a TensorFlow TensorProto file.

Supports Float64 tensors with shape [batch_size, feature_size] or [n].
Prefers tensor_content for efficient parsing, but still requires one copy to Vec<f64>.

# Byte Order
Assumes little-endian byte order (standard on x86_64).

# Returns
Tuple of `(flattened_data, num_samples, sample_size)`

# TODO
Add OOM protection for very large files

```rust
pub fn read_tensorflow_batch<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<(Vec<f64>, usize, usize)> { /* ... */ }
```

## Module `preprocessing`

```rust
pub mod preprocessing { /* ... */ }
```

### Types

#### Struct `Preprocessor`

Shared CPU-based pre-processing pipeline for quantum encoding.

Centralizes validation, normalization, and data preparation steps
to ensure consistency across different encoding strategies and backends.

```rust
pub struct Preprocessor;
```

##### Implementations

###### Methods

- ```rust
  pub fn validate_input(host_data: &[f64], num_qubits: usize) -> Result<()> { /* ... */ }
  ```
  Validates standard quantum input constraints.

- ```rust
  pub fn calculate_l2_norm(host_data: &[f64]) -> Result<f64> { /* ... */ }
  ```
  Calculates L2 norm of the input data in parallel on the CPU.

- ```rust
  pub fn validate_batch(batch_data: &[f64], num_samples: usize, sample_size: usize, num_qubits: usize) -> Result<()> { /* ... */ }
  ```
  Validates input constraints for batch processing.

- ```rust
  pub fn calculate_batch_l2_norms(batch_data: &[f64], _num_samples: usize, sample_size: usize) -> Result<Vec<f64>> { /* ... */ }
  ```
  Calculates L2 norms for a batch of samples in parallel.

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Module `reader`

Generic data reader interface for multiple input formats.

This module provides a trait-based architecture for reading quantum data
from various sources (Parquet, Arrow IPC, NumPy, PyTorch, etc.) in a
unified way without sacrificing performance or memory efficiency.

# Architecture

The reader system is based on two main traits:

- [`DataReader`]: Basic interface for batch reading
- [`StreamingDataReader`]: Extended interface for chunk-by-chunk streaming

# Example: Adding a New Format

To add support for a new format (e.g., NumPy):

```rust,ignore
use qdp_core::reader::{DataReader, Result};

pub struct NumpyReader {
    // format-specific fields
}

impl DataReader for NumpyReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        // implementation
    }
}
```

```rust
pub mod reader { /* ... */ }
```

### Traits

#### Trait `DataReader`

Generic data reader interface for batch quantum data.

Implementations should read data in the format:
- Flattened batch data (all samples concatenated)
- Number of samples
- Sample size (elements per sample)

This interface enables zero-copy streaming where possible and maintains
memory efficiency for large datasets.

```rust
pub trait DataReader {
    /* Associated items */
}
```

##### Required Items

###### Required Methods

- `read_batch`: Read all data from the source.

##### Provided Methods

- ```rust
  fn get_sample_size(self: &Self) -> Option<usize> { /* ... */ }
  ```
  Get the sample size if known before reading.

- ```rust
  fn get_num_samples(self: &Self) -> Option<usize> { /* ... */ }
  ```
  Get the total number of samples if known before reading.

##### Implementations

This trait is implemented for the following types:

- `ArrowIPCReader`
- `NumpyReader`
- `ParquetReader`
- `ParquetStreamingReader`
- `TensorFlowReader`
- `TorchReader`

#### Trait `StreamingDataReader`

Streaming data reader interface for large datasets.

This trait enables chunk-by-chunk reading for datasets that don't fit
in memory, maintaining constant memory usage regardless of file size.

```rust
pub trait StreamingDataReader: DataReader {
    /* Associated items */
}
```

##### Required Items

###### Required Methods

- `read_chunk`: Read a chunk of data into the provided buffer.
- `total_rows`: Get the total number of rows/samples in the data source.

##### Implementations

This trait is implemented for the following types:

- `ParquetStreamingReader`

## Module `readers`

Format-specific data reader implementations.

This module contains concrete implementations of the [`DataReader`] and
[`StreamingDataReader`] traits for various file formats.

# Fully Implemented Formats
- **Parquet**: [`ParquetReader`], [`ParquetStreamingReader`]
- **Arrow IPC**: [`ArrowIPCReader`]
- **NumPy**: [`NumpyReader`]
- **TensorFlow TensorProto**: [`TensorFlowReader`]
- **PyTorch**: [`TorchReader`] (feature: `pytorch`)

```rust
pub mod readers { /* ... */ }
```

### Modules

## Module `arrow_ipc`

Arrow IPC format reader implementation.

```rust
pub mod arrow_ipc { /* ... */ }
```

### Types

#### Struct `ArrowIPCReader`

Reader for Arrow IPC files containing FixedSizeList<Float64> or List<Float64> columns.

```rust
pub struct ArrowIPCReader {
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> { /* ... */ }
  ```
  Create a new Arrow IPC reader.

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **DataReader**
  - ```rust
    fn read_batch(self: &mut Self) -> Result<(Vec<f64>, usize, usize)> { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Module `numpy`

NumPy format reader implementation.

Provides support for reading .npy files containing 2D float64 arrays.

```rust
pub mod numpy { /* ... */ }
```

### Types

#### Struct `NumpyReader`

Reader for NumPy `.npy` files containing 2D float64 arrays.

# Expected Format
- 2D array with shape `[num_samples, sample_size]`
- Data type: `float64`
- Fortran (column-major) or C (row-major) order supported

# Example

```rust,ignore
use qdp_core::reader::DataReader;
use qdp_core::readers::NumpyReader;

let mut reader = NumpyReader::new("data.npy").unwrap();
let (data, num_samples, sample_size) = reader.read_batch().unwrap();
println!("Read {} samples of size {}", num_samples, sample_size);
```

```rust
pub struct NumpyReader {
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> { /* ... */ }
  ```
  Create a new NumPy reader.

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **DataReader**
  - ```rust
    fn read_batch(self: &mut Self) -> Result<(Vec<f64>, usize, usize)> { /* ... */ }
    ```

  - ```rust
    fn get_sample_size(self: &Self) -> Option<usize> { /* ... */ }
    ```

  - ```rust
    fn get_num_samples(self: &Self) -> Option<usize> { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Module `parquet`

Parquet format reader implementation.

```rust
pub mod parquet { /* ... */ }
```

### Types

#### Struct `ParquetReader`

Reader for Parquet files containing List<Float64> or FixedSizeList<Float64> columns.

```rust
pub struct ParquetReader {
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn new<P: AsRef<Path>>(path: P, batch_size: Option<usize>) -> Result<Self> { /* ... */ }
  ```
  Create a new Parquet reader.

###### Trait Implementations

- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **DataReader**
  - ```rust
    fn read_batch(self: &mut Self) -> Result<(Vec<f64>, usize, usize)> { /* ... */ }
    ```

  - ```rust
    fn get_sample_size(self: &Self) -> Option<usize> { /* ... */ }
    ```

  - ```rust
    fn get_num_samples(self: &Self) -> Option<usize> { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Struct `ParquetStreamingReader`

Streaming Parquet reader for List<Float64> and FixedSizeList<Float64> columns.

Reads Parquet files in chunks without loading entire file into memory.
Supports efficient streaming for large files via Producer-Consumer pattern.

```rust
pub struct ParquetStreamingReader {
    pub total_rows: usize,
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `total_rows` | `usize` |  |
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn new<P: AsRef<Path>>(path: P, batch_size: Option<usize>) -> Result<Self> { /* ... */ }
  ```
  Create a new streaming Parquet reader.

- ```rust
  pub fn get_sample_size(self: &Self) -> Option<usize> { /* ... */ }
  ```
  Get the sample size (number of elements per sample).

###### Trait Implementations

- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **DataReader**
  - ```rust
    fn read_batch(self: &mut Self) -> Result<(Vec<f64>, usize, usize)> { /* ... */ }
    ```

  - ```rust
    fn get_sample_size(self: &Self) -> Option<usize> { /* ... */ }
    ```

  - ```rust
    fn get_num_samples(self: &Self) -> Option<usize> { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **StreamingDataReader**
  - ```rust
    fn read_chunk(self: &mut Self, buffer: &mut [f64]) -> Result<usize> { /* ... */ }
    ```

  - ```rust
    fn total_rows(self: &Self) -> usize { /* ... */ }
    ```

- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Module `tensorflow`

TensorFlow TensorProto format reader implementation.

```rust
pub mod tensorflow { /* ... */ }
```

### Types

#### Struct `TensorFlowReader`

Reader for TensorFlow TensorProto files.

Supports Float64 tensors with shape [batch_size, feature_size] or [n].
Prefers tensor_content for efficient parsing, but still requires one copy to Vec<f64>.

# Byte Order
This implementation assumes little-endian byte order, which is the standard
on x86_64 platforms. TensorFlow typically uses host byte order.

```rust
pub struct TensorFlowReader {
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> { /* ... */ }
  ```
  Create a new TensorFlow reader from a file path.

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **DataReader**
  - ```rust
    fn read_batch(self: &mut Self) -> Result<(Vec<f64>, usize, usize)> { /* ... */ }
    ```

  - ```rust
    fn get_sample_size(self: &Self) -> Option<usize> { /* ... */ }
    ```

  - ```rust
    fn get_num_samples(self: &Self) -> Option<usize> { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Module `torch`

PyTorch tensor reader implementation.

Supports `.pt`/`.pth` files containing a single tensor saved with `torch.save`.
The tensor must be 1D or 2D and will be converted to `float64`.
Requires the `pytorch` feature to be enabled.

```rust
pub mod torch { /* ... */ }
```

### Types

#### Struct `TorchReader`

Reader for PyTorch `.pt`/`.pth` tensor files.

```rust
pub struct TorchReader {
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> { /* ... */ }
  ```
  Create a new PyTorch reader.

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **DataReader**
  - ```rust
    fn read_batch(self: &mut Self) -> Result<(Vec<f64>, usize, usize)> { /* ... */ }
    ```

  - ```rust
    fn get_sample_size(self: &Self) -> Option<usize> { /* ... */ }
    ```

  - ```rust
    fn get_num_samples(self: &Self) -> Option<usize> { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
### Re-exports

#### Re-export `ArrowIPCReader`

```rust
pub use arrow_ipc::ArrowIPCReader;
```

#### Re-export `NumpyReader`

```rust
pub use numpy::NumpyReader;
```

#### Re-export `ParquetReader`

```rust
pub use parquet::ParquetReader;
```

#### Re-export `ParquetStreamingReader`

```rust
pub use parquet::ParquetStreamingReader;
```

#### Re-export `TensorFlowReader`

```rust
pub use tensorflow::TensorFlowReader;
```

#### Re-export `TorchReader`

```rust
pub use torch::TorchReader;
```

## Module `tf_proto`

TensorFlow TensorProto protobuf definitions.

This module contains the generated protobuf code for TensorFlow TensorProto format.
The code is generated at build time by prost-build from `proto/tensor.proto`.

```rust
pub mod tf_proto { /* ... */ }
```

### Modules

## Module `tensorflow`

```rust
pub mod tensorflow { /* ... */ }
```

### Types

#### Struct `TensorProto`

**Attributes:**

- `Other("#[allow(clippy::derive_partial_eq_without_eq)]")`

TensorProto - only define necessary fields, field numbers match TensorFlow official

```rust
pub struct TensorProto {
    pub dtype: i32,
    pub tensor_shape: ::core::option::Option<TensorShapeProto>,
    pub tensor_content: ::prost::bytes::Bytes,
    pub double_val: ::prost::alloc::vec::Vec<f64>,
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `dtype` | `i32` | Field 1: dtype (enum DataType in TF, but varint in wire format)<br>DT_DOUBLE = 2 (see tensorflow/core/framework/types.proto) |
| `tensor_shape` | `::core::option::Option<TensorShapeProto>` | Field 2: tensor_shape |
| `tensor_content` | `::prost::bytes::Bytes` | Field 4: tensor_content (preferred for efficient parsing) |
| `double_val` | `::prost::alloc::vec::Vec<f64>` | Field 6: double_val (fallback, only used when tensor_content is empty) |

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Clone**
  - ```rust
    fn clone(self: &Self) -> TensorProto { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut ::core::fmt::Formatter<''_>) -> ::core::fmt::Result { /* ... */ }
    ```

- **Default**
  - ```rust
    fn default() -> Self { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Message**
  - ```rust
    fn encoded_len(self: &Self) -> usize { /* ... */ }
    ```

  - ```rust
    fn clear(self: &mut Self) { /* ... */ }
    ```

- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &TensorProto) -> bool { /* ... */ }
    ```

- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Struct `TensorShapeProto`

**Attributes:**

- `Other("#[allow(clippy::derive_partial_eq_without_eq)]")`

```rust
pub struct TensorShapeProto {
    pub dim: ::prost::alloc::vec::Vec<Dim>,
    pub unknown_rank: bool,
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `dim` | `::prost::alloc::vec::Vec<Dim>` | Field 2: dim (field number matches official) |
| `unknown_rank` | `bool` | Field 3: unknown_rank (optional; helps with better error messages) |

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Clone**
  - ```rust
    fn clone(self: &Self) -> TensorShapeProto { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut ::core::fmt::Formatter<''_>) -> ::core::fmt::Result { /* ... */ }
    ```

- **Default**
  - ```rust
    fn default() -> Self { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Message**
  - ```rust
    fn encoded_len(self: &Self) -> usize { /* ... */ }
    ```

  - ```rust
    fn clear(self: &mut Self) { /* ... */ }
    ```

- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &TensorShapeProto) -> bool { /* ... */ }
    ```

- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
#### Struct `Dim`

**Attributes:**

- `Other("#[allow(clippy::derive_partial_eq_without_eq)]")`

```rust
pub struct Dim {
    pub size: i64,
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `size` | `i64` | Field 1: size<br><br>Skip name field (field number 2) to reduce parsing overhead |

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Clone**
  - ```rust
    fn clone(self: &Self) -> Dim { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut ::core::fmt::Formatter<''_>) -> ::core::fmt::Result { /* ... */ }
    ```

- **Default**
  - ```rust
    fn default() -> Self { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Message**
  - ```rust
    fn encoded_len(self: &Self) -> usize { /* ... */ }
    ```

  - ```rust
    fn clear(self: &mut Self) { /* ... */ }
    ```

- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &Dim) -> bool { /* ... */ }
    ```

- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Types

### Struct `QdpEngine`

Main entry point for Mahout QDP

Manages GPU context and dispatches encoding tasks.
Provides unified interface for device management, memory allocation, and DLPack.

```rust
pub struct QdpEngine {
    // Some fields omitted
}
```

#### Fields

| Name | Type | Documentation |
|------|------|---------------|
| *private fields* | ... | *Some fields have been omitted* |

#### Implementations

##### Methods

- ```rust
  pub fn new(device_id: usize) -> Result<Self> { /* ... */ }
  ```
  Initialize engine on GPU device

- ```rust
  pub fn new_with_precision(device_id: usize, precision: Precision) -> Result<Self> { /* ... */ }
  ```
  Initialize engine with explicit precision.

- ```rust
  pub fn encode(self: &Self, data: &[f64], num_qubits: usize, encoding_method: &str) -> Result<*mut DLManagedTensor> { /* ... */ }
  ```
  Encode classical data into quantum state

- ```rust
  pub fn device(self: &Self) -> &CudaDevice { /* ... */ }
  ```
  Get CUDA device reference for advanced operations

- ```rust
  pub fn encode_batch(self: &Self, batch_data: &[f64], num_samples: usize, sample_size: usize, num_qubits: usize, encoding_method: &str) -> Result<*mut DLManagedTensor> { /* ... */ }
  ```
  Encode multiple samples in a single fused kernel (most efficient)

- ```rust
  pub fn encode_from_parquet(self: &Self, path: &str, num_qubits: usize, encoding_method: &str) -> Result<*mut DLManagedTensor> { /* ... */ }
  ```
  Streaming Parquet encoder with multi-threaded IO

- ```rust
  pub fn encode_from_arrow_ipc(self: &Self, path: &str, num_qubits: usize, encoding_method: &str) -> Result<*mut DLManagedTensor> { /* ... */ }
  ```
  Load data from Arrow IPC file and encode into quantum state

- ```rust
  pub fn encode_from_numpy(self: &Self, path: &str, num_qubits: usize, encoding_method: &str) -> Result<*mut DLManagedTensor> { /* ... */ }
  ```
  Load data from NumPy .npy file and encode into quantum state

- ```rust
  pub fn encode_from_torch(self: &Self, path: &str, num_qubits: usize, encoding_method: &str) -> Result<*mut DLManagedTensor> { /* ... */ }
  ```
  Load data from PyTorch .pt/.pth file and encode into quantum state

- ```rust
  pub fn encode_from_tensorflow(self: &Self, path: &str, num_qubits: usize, encoding_method: &str) -> Result<*mut DLManagedTensor> { /* ... */ }
  ```
  Load data from TensorFlow TensorProto file and encode into quantum state

##### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **Clone**
  - ```rust
    fn clone(self: &Self) -> QdpEngine { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **IntoEither**
- **Pointable**
  - ```rust
    unsafe fn init(init: <T as Pointable>::Init) -> usize { /* ... */ }
    ```

  - ```rust
    unsafe fn deref<''a>(ptr: usize) -> &''a T { /* ... */ }
    ```

  - ```rust
    unsafe fn deref_mut<''a>(ptr: usize) -> &''a mut T { /* ... */ }
    ```

  - ```rust
    unsafe fn drop(ptr: usize) { /* ... */ }
    ```

- **RefUnwindSafe**
- **Send**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnwindSafe**
## Macros

### Macro `profile_scope`

**Attributes:**

- `Other("#[attr = CfgTrace([Not(NameValue { name: \"feature\", value: Some(\"observability\"), span: qdp-core/src/profiling.rs:46:11: 46:36 (#0) }, qdp-core/src/profiling.rs:46:10: 46:37 (#0))])]")`
- `MacroExport`

No-op version when observability is disabled

Compiler eliminates this completely, zero runtime cost.

```rust
pub macro_rules! profile_scope {
    /* macro_rules! profile_scope {
    ($name:expr) => { ... };
} */
}
```

### Macro `profile_mark`

**Attributes:**

- `Other("#[attr = CfgTrace([Not(NameValue { name: \"feature\", value: Some(\"observability\"), span: qdp-core/src/profiling.rs:71:11: 71:36 (#0) }, qdp-core/src/profiling.rs:71:10: 71:37 (#0))])]")`
- `MacroExport`

No-op version when observability is disabled

```rust
pub macro_rules! profile_mark {
    /* macro_rules! profile_mark {
    ($name:expr) => { ... };
} */
}
```

## Re-exports

### Re-export `MahoutError`

```rust
pub use error::MahoutError;
```

### Re-export `Result`

```rust
pub use error::Result;
```

### Re-export `Precision`

```rust
pub use gpu::memory::Precision;
```

### Re-export `QuantumEncoder`

```rust
pub use gpu::QuantumEncoder;
```

