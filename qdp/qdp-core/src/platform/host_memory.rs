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

//! Host (CPU) memory queries used by readers to refuse loads that would OOM.

/// Bytes of host memory the kernel estimates are available for new
/// allocations without swapping (`MemAvailable` in `/proc/meminfo`).
///
/// Returns `None` when the value cannot be determined: on non-Linux targets,
/// when `/proc/meminfo` is unreadable, or on kernels older than 3.14 that do
/// not report `MemAvailable`. Callers must treat `None` as "unknown", not as
/// "zero".
///
/// Note: inside a container this reports the host's view, not a cgroup
/// memory limit.
pub(crate) fn available_host_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_mem_available(&meminfo)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Extract `MemAvailable` (reported in kB) from `/proc/meminfo` contents and
/// convert it to bytes. Pure so it can be unit-tested on any platform.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_mem_available(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        let mut parts = rest.split_whitespace();
        let value: u64 = parts.next()?.parse().ok()?;
        // The kernel always reports this field in kB (KiB).
        match parts.next() {
            Some("kB") | None => value.checked_mul(1024),
            Some(_) => None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mem_available_in_kib() {
        let sample = "MemTotal:       16384000 kB\n\
                      MemFree:         1000000 kB\n\
                      MemAvailable:    6170172 kB\n\
                      Buffers:          123456 kB\n";
        assert_eq!(parse_mem_available(sample), Some(6_170_172 * 1024));
    }

    #[test]
    fn missing_mem_available_is_none() {
        // Kernels before 3.14 do not report MemAvailable.
        let sample = "MemTotal:       16384000 kB\nMemFree:         1000000 kB\n";
        assert_eq!(parse_mem_available(sample), None);
    }

    #[test]
    fn malformed_mem_available_is_none() {
        assert_eq!(parse_mem_available("MemAvailable: lots kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: 12 MB\n"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn reads_live_value_on_linux() {
        // Every supported Linux kernel exposes MemAvailable; it is never 0
        // on a running system.
        let bytes = available_host_memory_bytes().expect("MemAvailable on Linux");
        assert!(bytes > 0);
    }
}
