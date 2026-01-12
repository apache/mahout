use crate::dlpack::DLManagedTensor;
use crate::platform::fallback;
use crate::{QdpEngine, Result};

pub(crate) fn encode_from_parquet(
    engine: &QdpEngine,
    path: &str,
    num_qubits: usize,
    encoding_method: &str,
) -> Result<*mut DLManagedTensor> {
    fallback::encode_from_parquet(engine, path, num_qubits, encoding_method)
}
