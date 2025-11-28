use thiserror::Error;

/// Error types for Mahout QDP operations
#[derive(Error, Debug)]
pub enum MahoutError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("Kernel launch failed: {0}")]
    KernelLaunch(String),

    #[error("DLPack operation failed: {0}")]
    DLPack(String),
}

/// Result type alias for Mahout operations
pub type Result<T> = std::result::Result<T, MahoutError>;

