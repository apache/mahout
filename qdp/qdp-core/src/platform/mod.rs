#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "windows")]
mod windows;
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
mod other;
mod fallback;

#[cfg(target_os = "linux")]
pub(crate) use linux::encode_from_parquet;
#[cfg(target_os = "windows")]
pub(crate) use windows::encode_from_parquet;
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub(crate) use other::encode_from_parquet;
