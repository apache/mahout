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

use crate::gpu::topology::GpuTopology;

pub(crate) fn policy_device_ids(
    topology: &GpuTopology,
    shard_device_ids: impl IntoIterator<Item = usize>,
) -> (Option<usize>, Vec<usize>) {
    let shard_device_ids = shard_device_ids.into_iter().collect::<Vec<_>>();
    let recommended_gather_device_id = topology
        .preferred_gather_index()
        .and_then(|idx| shard_device_ids.get(idx).copied());
    let recommended_placement_device_ids = topology
        .recommended_placement_order()
        .into_iter()
        .filter_map(|idx| shard_device_ids.get(idx).copied())
        .collect();

    (
        recommended_gather_device_id,
        recommended_placement_device_ids,
    )
}

#[cfg(test)]
mod tests {
    use super::policy_device_ids;
    use crate::gpu::topology::{GpuTopology, LinkKind};

    #[test]
    fn policy_device_ids_skips_out_of_range_topology_indices() {
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

        let (gather, placement) = policy_device_ids(&topology, [17, 23]);

        assert_eq!(gather, Some(23));
        assert_eq!(placement, vec![23, 17]);
    }
}
