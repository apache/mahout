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

"""Unified tensor facade for backend-native QDP results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QdpTensor:
    """Thin DLPack facade over backend-native tensor producers."""

    value: Any
    backend: str

    def __dlpack__(self, stream: int | None = None) -> Any:
        """Return a DLPack capsule for the wrapped backend tensor.

        :param stream: Optional consumer stream to pass through to the wrapped
            tensor's ``__dlpack__`` implementation.
        :returns: A DLPack capsule representing ``value``.
        :raises RuntimeError: If the wrapped value does not implement
            ``__dlpack__``.
        """
        if not hasattr(self.value, "__dlpack__"):
            raise RuntimeError(
                f"Backend '{self.backend}' returned object without __dlpack__ support: "
                f"{type(self.value)!r}"
            )
        if stream is None:
            return self.value.__dlpack__()
        return self.value.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> Any:
        """Return the DLPack device descriptor for the wrapped tensor.

        :returns: The ``(device_type, device_id)`` tuple reported by ``value``.
        :raises RuntimeError: If the wrapped value does not implement
            ``__dlpack_device__``.
        """
        if not hasattr(self.value, "__dlpack_device__"):
            raise RuntimeError(
                f"Backend '{self.backend}' returned object without __dlpack_device__ support: "
                f"{type(self.value)!r}"
            )
        return self.value.__dlpack_device__()

    def to_torch(self) -> Any:
        """Convert the wrapped tensor to a PyTorch tensor via DLPack.

        :returns: A ``torch.Tensor`` sharing storage with the backend tensor
            when the backend's DLPack producer supports zero-copy exchange.
        """
        import torch

        return torch.from_dlpack(self)


QuantumTensor = QdpTensor
