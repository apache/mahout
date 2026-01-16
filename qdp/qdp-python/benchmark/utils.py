#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared utility functions for QDP benchmarks.

This module provides common data generation and normalization functions
used across multiple benchmark scripts.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import torch


def build_sample(
    seed: int, vector_len: int, encoding_method: str = "amplitude"
) -> np.ndarray:
    """
    Build a single sample vector for benchmarking.

    Args:
        seed: Seed value used to generate deterministic data.
        vector_len: Length of the vector (2^num_qubits for amplitude encoding).
        encoding_method: Either "amplitude" or "basis".

    Returns:
        NumPy array containing the sample data.
    """
    if encoding_method == "basis":
        # Basis encoding: single index per sample
        mask = np.uint64(vector_len - 1)
        idx = np.uint64(seed) & mask
        return np.array([idx], dtype=np.float64)
    else:
        # Amplitude encoding: full vector
        mask = np.uint64(vector_len - 1)
        scale = 1.0 / vector_len
        idx = np.arange(vector_len, dtype=np.uint64)
        mixed = (idx + np.uint64(seed)) & mask
        return mixed.astype(np.float64) * scale


def generate_batch_data(
    n_samples: int,
    dim: int,
    encoding_method: str = "amplitude",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate batch data for benchmarking.

    Args:
        n_samples: Number of samples to generate.
        dim: Dimension of each sample (2^num_qubits for amplitude encoding).
        encoding_method: Either "amplitude" or "basis".
        seed: Random seed for reproducibility.

    Returns:
        NumPy array of shape (n_samples, dim) for amplitude encoding
        or (n_samples, 1) for basis encoding.
    """
    np.random.seed(seed)
    if encoding_method == "basis":
        # Basis encoding: single index per sample
        return np.random.randint(0, dim, size=(n_samples, 1)).astype(np.float64)
    else:
        # Amplitude encoding: full vectors
        return np.random.rand(n_samples, dim).astype(np.float64)


def normalize_batch(
    batch: np.ndarray, encoding_method: str = "amplitude"
) -> np.ndarray:
    """
    Normalize a batch of vectors (L2 normalization).

    Args:
        batch: NumPy array of shape (batch_size, vector_len).
        encoding_method: Either "amplitude" or "basis".

    Returns:
        Normalized batch. For basis encoding, returns the input unchanged.
    """
    if encoding_method == "basis":
        # Basis encoding doesn't need normalization (indices)
        return batch
    # Amplitude encoding: normalize vectors
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return batch / norms


def normalize_batch_torch(
    batch: torch.Tensor, encoding_method: str = "amplitude"
) -> torch.Tensor:
    """
    Normalize a batch of PyTorch tensors (L2 normalization).

    Args:
        batch: PyTorch tensor of shape (batch_size, vector_len).
        encoding_method: Either "amplitude" or "basis".

    Returns:
        Normalized batch. For basis encoding, returns the input unchanged.
    """
    if encoding_method == "basis":
        # Basis encoding doesn't need normalization (indices)
        return batch
    # Amplitude encoding: normalize vectors
    norms = torch.norm(batch, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-10)  # Avoid division by zero
    return batch / norms


def prefetched_batches(
    total_batches: int,
    batch_size: int,
    vector_len: int,
    prefetch: int,
    encoding_method: str = "amplitude",
):
    """
    Generate prefetched batches of NumPy arrays for benchmarking.

    Uses a background thread to prefetch batches and keep the GPU fed.

    Args:
        total_batches: Total number of batches to generate.
        batch_size: Number of samples per batch.
        vector_len: Length of each vector (2^num_qubits).
        prefetch: Number of batches to prefetch.
        encoding_method: Either "amplitude" or "basis".

    Yields:
        NumPy arrays of shape (batch_size, vector_len) or (batch_size, 1).
    """
    q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=prefetch)

    def producer():
        for batch_idx in range(total_batches):
            base = batch_idx * batch_size
            batch = [
                build_sample(base + i, vector_len, encoding_method)
                for i in range(batch_size)
            ]
            q.put(np.stack(batch))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch


def prefetched_batches_torch(
    total_batches: int,
    batch_size: int,
    vector_len: int,
    prefetch: int,
    encoding_method: str = "amplitude",
):
    """
    Generate prefetched batches as PyTorch tensors for benchmarking.

    Uses a background thread to prefetch batches and keep the GPU fed.

    Args:
        total_batches: Total number of batches to generate.
        batch_size: Number of samples per batch.
        vector_len: Length of each vector (2^num_qubits).
        prefetch: Number of batches to prefetch.
        encoding_method: Either "amplitude" or "basis".

    Yields:
        PyTorch tensors of shape (batch_size, vector_len) or (batch_size, 1).
    """
    q: queue.Queue[torch.Tensor | None] = queue.Queue(maxsize=prefetch)

    def producer():
        for batch_idx in range(total_batches):
            base = batch_idx * batch_size
            batch = [
                build_sample(base + i, vector_len, encoding_method)
                for i in range(batch_size)
            ]
            # Convert to PyTorch tensor (CPU, float64, contiguous)
            batch_tensor = torch.tensor(
                np.stack(batch), dtype=torch.float64, device="cpu"
            )
            assert batch_tensor.is_contiguous(), "Tensor should be contiguous"
            q.put(batch_tensor)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch
