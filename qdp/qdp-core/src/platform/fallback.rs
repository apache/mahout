use crate::dlpack::DLManagedTensor;
use crate::{QdpEngine, Result};

pub(crate) fn encode_from_parquet(
    engine: &QdpEngine,
    path: &str,
    num_qubits: usize,
    encoding_method: &str,
) -> Result<*mut DLManagedTensor> {
    crate::profile_scope!("Mahout::EncodeFromParquet");

    let (batch_data, num_samples, sample_size) = crate::io::read_parquet_batch(path)?;
    engine.encode_batch(
        &batch_data,
        num_samples,
        sample_size,
        num_qubits,
        encoding_method,
    )
}
