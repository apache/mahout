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

use qdp_core::{DeviceMesh, GpuTopology, LinkKind};

#[test]
fn placeholder_topology_marks_self_edges() {
    let topology = GpuTopology::placeholder(2);
    assert!(topology.peer_access[0][0]);
    assert!(topology.peer_access[1][1]);
    assert_eq!(topology.links[0][0], LinkKind::SameDevice);
    assert_eq!(topology.links[1][1], LinkKind::SameDevice);
    assert!(!topology.peer_access[0][1]);
    assert_eq!(topology.links[0][1], LinkKind::Unknown);
}

#[test]
fn duplicate_device_ids_are_rejected() {
    let err = DeviceMesh::new(vec![0, 0]).err().unwrap();
    assert!(matches!(err, qdp_core::MahoutError::InvalidInput(_)));
}

#[test]
fn power_of_two_validation_rejects_three_devices() {
    let topology = GpuTopology::placeholder(3);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1, 2],
        devices: Vec::new(),
        topology,
    };
    let err = mesh.validate_power_of_two().unwrap_err();
    assert!(matches!(err, qdp_core::MahoutError::InvalidInput(_)));
}

#[test]
#[cfg(target_os = "linux")]
fn from_parts_rejects_device_handle_mismatch() {
    let device = match cudarc::driver::CudaDevice::new(0) {
        Ok(device) => device,
        Err(_) => return,
    };
    let topology = GpuTopology::placeholder(1);
    let err = DeviceMesh::from_parts(vec![1], vec![device], topology)
        .err()
        .unwrap();
    assert!(matches!(err, qdp_core::MahoutError::InvalidInput(_)));
}

#[test]
fn preferred_gather_index_prefers_best_connected_device() {
    let topology = GpuTopology {
        peer_access: vec![
            vec![true, true, false],
            vec![true, true, true],
            vec![false, true, true],
        ],
        links: vec![
            vec![LinkKind::SameDevice, LinkKind::Pix, LinkKind::Unknown],
            vec![LinkKind::Pix, LinkKind::SameDevice, LinkKind::Node],
            vec![LinkKind::Unknown, LinkKind::Node, LinkKind::SameDevice],
        ],
    };

    assert_eq!(topology.preferred_gather_index(), Some(1));
    assert_eq!(topology.recommended_placement_order(), vec![1, 0, 2]);
}
