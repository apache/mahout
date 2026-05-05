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

use std::time::Instant;

use qdp_core::{DistributionMode, MahoutError};
use qdp_core::{HostCommunicator, PlacementRequest, Precision, QdpEngine, ShardPolicy};

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn parse_device_ids() -> Result<Vec<usize>, MahoutError> {
    let raw = std::env::var("GPU_IDS").unwrap_or_else(|_| "0,1,2,3,4,5".to_string());
    let mut ids = Vec::new();
    for piece in raw.split(',') {
        let trimmed = piece.trim();
        if trimmed.is_empty() {
            continue;
        }
        ids.push(trimmed.parse::<usize>().map_err(|err| {
            MahoutError::InvalidInput(format!("Invalid GPU ID '{trimmed}': {err}"))
        })?);
    }

    if ids.is_empty() {
        return Err(MahoutError::InvalidInput(
            "GPU_IDS must contain at least one CUDA device ID".to_string(),
        ));
    }

    Ok(ids)
}

fn main() -> Result<(), MahoutError> {
    let num_qubits = std::env::var("QUBITS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(34);
    let host_len = std::env::var("HOST_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let precision = match std::env::var("PRECISION").ok().as_deref() {
        Some("f64") | Some("float64") => Precision::Float64,
        _ => Precision::Float32,
    };
    let shard_policy = match std::env::var("SHARD_POLICY").ok().as_deref() {
        Some("equal") => ShardPolicy::Equal,
        _ => ShardPolicy::BalancedUneven,
    };
    let device_ids = parse_device_ids()?;
    let request =
        PlacementRequest::new(num_qubits, DistributionMode::ShardedCapacity, shard_policy);
    let communicator = HostCommunicator;
    let host_data = vec![1.0f64; host_len];

    println!(
        "Starting MPI-shaped distributed amplitude probe: qubits={}, host_len={}, gpus={:?}, precision={:?}, shard_policy={:?}, communicator=host-loopback",
        num_qubits, host_len, device_ids, precision, shard_policy
    );

    let prepare_start = Instant::now();
    let prepared = QdpEngine::prepare_distributed_amplitude_with_communicator(
        device_ids.clone(),
        &host_data,
        num_qubits,
        precision,
        Some(request.clone()),
        &communicator,
    )?;
    let prepare_elapsed = prepare_start.elapsed();

    println!(
        "Prepared in {:.3}s; global_len={}; shards={}; max_local_len={}; estimated_max_shard_gib={:.2}; gather_device={:?}",
        prepare_elapsed.as_secs_f64(),
        prepared.plan.global_len,
        prepared.layout.num_shards(),
        prepared.plan.max_local_len(),
        gib(prepared.plan.estimated_max_shard_bytes(precision)?),
        prepared.layout.recommended_gather_device_id()
    );

    for shard in &prepared.layout.shards {
        let shard_bytes = match precision {
            Precision::Float32 => shard.local_len * 8,
            Precision::Float64 => shard.local_len * 16,
        };
        println!(
            "  shard {} -> cuda:{} range=[{}, {}) local_len={} (~{:.2} GiB)",
            shard.shard_id,
            shard.device_id,
            shard.start_idx,
            shard.end_idx,
            shard.local_len,
            gib(shard_bytes)
        );
    }

    let encode_start = Instant::now();
    let state = QdpEngine::encode_distributed_amplitude_to_shards_with_communicator(
        device_ids,
        &host_data,
        num_qubits,
        precision,
        Some(request),
        &communicator,
    )?;
    let encode_elapsed = encode_start.elapsed();

    println!(
        "Encoded in {:.3}s; state_shards={}; placement={:?}",
        encode_elapsed.as_secs_f64(),
        state.num_shards(),
        state.recommended_placement_device_ids()
    );

    Ok(())
}
