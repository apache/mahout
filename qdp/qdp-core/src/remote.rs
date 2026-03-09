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

//! Remote I/O support for cloud object storage.
//!
//! When the `remote-io` feature is enabled, cloud URLs (currently `s3://`)
//! are transparently downloaded to a local temp file before being passed to
//! readers. Adding GCS (`gs://`) or Azure (`az://`) support requires only a
//! new match arm in [`build_store`] and the corresponding `object_store`
//! cargo feature.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::path::Path as ObjectPath;
use tempfile::NamedTempFile;

use crate::error::{MahoutError, Result};

/// Recognized cloud URL schemes.
const REMOTE_SCHEMES: &[&str] = &["s3://"];

/// Returns true if `path` is a recognized remote URL.
pub fn is_remote_path(path: &str) -> bool {
    REMOTE_SCHEMES.iter().any(|s| path.starts_with(s))
}

/// Parse a cloud URL into (scheme, bucket, key).
fn parse_url(url: &str) -> Result<(&str, &str, &str)> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| MahoutError::InvalidInput(format!("Not a remote URL: {}", url)))?;
    let (bucket, key) = rest.split_once('/').ok_or_else(|| {
        MahoutError::InvalidInput(format!(
            "Remote URL must have the form scheme://bucket/key, got: {}",
            url
        ))
    })?;
    if bucket.is_empty() || key.is_empty() {
        return Err(MahoutError::InvalidInput(format!(
            "Remote URL has empty bucket or key: {}",
            url
        )));
    }
    Ok((scheme, bucket, key))
}

/// Build an [`ObjectStore`] for the given scheme and bucket.
/// Add new match arms here to support additional cloud providers.
fn build_store(scheme: &str, bucket: &str) -> Result<Arc<dyn ObjectStore>> {
    match scheme {
        "s3" => {
            let store = object_store::aws::AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()
                .map_err(|e| {
                    MahoutError::Io(format!(
                        "Failed to build S3 client for bucket '{}': {}",
                        bucket, e
                    ))
                })?;
            Ok(Arc::new(store))
        }
        // To add GCS:  "gs" => { ... GoogleCloudStorageBuilder ... }
        // To add Azure: "az" | "abfs" => { ... MicrosoftAzureBuilder ... }
        _ => Err(MahoutError::InvalidInput(format!(
            "Unsupported remote scheme '{}://'. Supported: {}",
            scheme,
            REMOTE_SCHEMES.join(", ")
        ))),
    }
}

/// Holds the temp file so it is not deleted while still needed.
/// Drop this value to clean up the temp file.
pub struct ResolvedPath {
    /// Local path to the (possibly downloaded) file.
    pub path: PathBuf,
    /// Keeps the temp file alive; None for local paths.
    _tempfile: Option<NamedTempFile>,
}

impl AsRef<Path> for ResolvedPath {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

/// If `path` is a remote URL, download it to a local temp file and return
/// the temp path. Otherwise return the path as-is. The returned
/// [`ResolvedPath`] keeps the temp file alive; dropping it removes the file.
pub fn resolve_path(path: &str) -> Result<ResolvedPath> {
    if !is_remote_path(path) {
        return Ok(ResolvedPath {
            path: PathBuf::from(path),
            _tempfile: None,
        });
    }
    download_to_tempfile(path)
}

/// Download a remote object to a local temp file, streaming chunks to disk.
fn download_to_tempfile(url: &str) -> Result<ResolvedPath> {
    use futures::TryStreamExt;

    let (scheme, bucket, key) = parse_url(url)?;
    let store = build_store(scheme, bucket)?;
    let object_path = ObjectPath::from(key);

    // Preserve the original extension so downstream readers dispatch correctly.
    let extension = Path::new(key)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("tmp");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| MahoutError::Io(format!("Failed to create tokio runtime: {}", e)))?;

    let mut tmpfile = tempfile::Builder::new()
        .suffix(&format!(".{}", extension))
        .tempfile()
        .map_err(|e| MahoutError::Io(format!("Failed to create temp file: {}", e)))?;

    rt.block_on(async {
        let result = store
            .get(&object_path)
            .await
            .map_err(|e| MahoutError::Io(format!("Failed to download {}: {}", url, e)))?;
        let mut stream = result.into_stream();
        while let Some(chunk) = stream
            .try_next()
            .await
            .map_err(|e| MahoutError::Io(format!("Failed to read chunk from {}: {}", url, e)))?
        {
            tmpfile
                .write_all(&chunk)
                .map_err(|e| MahoutError::Io(format!("Failed to write temp file: {}", e)))?;
        }
        Ok::<(), MahoutError>(())
    })?;

    tmpfile
        .flush()
        .map_err(|e| MahoutError::Io(format!("Failed to flush temp file: {}", e)))?;

    let path = tmpfile.path().to_path_buf();
    Ok(ResolvedPath {
        path,
        _tempfile: Some(tmpfile),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_remote_path() {
        assert!(is_remote_path("s3://bucket/key.parquet"));
        assert!(!is_remote_path("/tmp/local.parquet"));
        assert!(!is_remote_path("data.parquet"));
    }

    #[test]
    fn test_parse_url_s3() {
        let (scheme, bucket, key) = parse_url("s3://my-bucket/path/to/data.parquet").unwrap();
        assert_eq!(scheme, "s3");
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/data.parquet");
    }

    #[test]
    fn test_parse_url_no_key() {
        assert!(parse_url("s3://bucket-only").is_err());
    }

    #[test]
    fn test_parse_url_empty_parts() {
        assert!(parse_url("s3:///key").is_err());
        assert!(parse_url("s3://bucket/").is_err());
    }

    #[test]
    fn test_unsupported_scheme() {
        let err = build_store("gcs", "bucket").unwrap_err();
        assert!(err.to_string().contains("Unsupported remote scheme"));
    }

    #[test]
    fn test_resolve_local_path() {
        let resolved = resolve_path("/tmp/local.parquet").unwrap();
        assert_eq!(resolved.path, PathBuf::from("/tmp/local.parquet"));
    }

    /// Integration test: download from a real S3-compatible endpoint (MinIO).
    ///
    /// Requires a running MinIO with a test file uploaded. Skipped unless
    /// `MAHOUT_TEST_S3` env var is set. Run with:
    ///
    /// ```sh
    /// MAHOUT_TEST_S3=1 \
    /// AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
    /// AWS_ENDPOINT=http://localhost:9123 AWS_REGION=us-east-1 AWS_ALLOW_HTTP=true \
    /// cargo test -p qdp-core --features remote-io -- test_download_from_minio
    /// ```
    #[test]
    fn test_download_from_minio() {
        if std::env::var("MAHOUT_TEST_S3").is_err() {
            eprintln!("skipping test_download_from_minio (set MAHOUT_TEST_S3=1 to run)");
            return;
        }
        let resolved = resolve_path("s3://test-bucket/data.parquet").unwrap();
        assert!(
            resolved.path.exists(),
            "temp file should exist after download"
        );
        assert!(
            resolved.path.to_string_lossy().ends_with(".parquet"),
            "temp file should preserve .parquet extension"
        );
        let file_len = std::fs::metadata(&resolved.path).unwrap().len();
        assert!(file_len > 0, "downloaded file should not be empty");

        // Verify it's a valid parquet that our reader can parse.
        use crate::reader::DataReader;
        let mut reader = crate::readers::ParquetReader::new(
            &resolved.path,
            None,
            crate::reader::NullHandling::FillZero,
        )
        .expect("ParquetReader should open downloaded file");
        let (data, num_samples, sample_size) =
            reader.read_batch().expect("read_batch should succeed");
        assert_eq!(num_samples, 8);
        assert_eq!(sample_size, 4);
        assert_eq!(data.len(), 32);
    }
}
