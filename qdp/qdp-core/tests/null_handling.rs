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

//! Tests for the unified NullHandling policy.

use arrow::array::Float64Array;
use qdp_core::reader::{NullHandling, handle_float64_nulls};

#[test]
fn fill_zero_replaces_nulls() {
    let array = Float64Array::from(vec![Some(1.0), None, Some(3.0), None]);
    let mut output = Vec::new();
    handle_float64_nulls(&mut output, &array, NullHandling::FillZero).unwrap();
    assert_eq!(output, vec![1.0, 0.0, 3.0, 0.0]);
}

#[test]
fn reject_returns_error_on_null() {
    let array = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let mut output = Vec::new();
    let result = handle_float64_nulls(&mut output, &array, NullHandling::Reject);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Null value encountered"),
        "unexpected error: {}",
        err_msg
    );
}

#[test]
fn no_nulls_fast_path() {
    let array = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
    let mut output = Vec::new();

    // Both policies should succeed and produce the same result when no nulls present
    handle_float64_nulls(&mut output, &array, NullHandling::FillZero).unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);

    let mut output2 = Vec::new();
    handle_float64_nulls(&mut output2, &array, NullHandling::Reject).unwrap();
    assert_eq!(output2, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn default_is_fill_zero() {
    assert_eq!(NullHandling::default(), NullHandling::FillZero);
}

#[test]
fn fill_zero_on_all_nulls() {
    let array = Float64Array::from(vec![None, None, None]);
    let mut output = Vec::new();
    handle_float64_nulls(&mut output, &array, NullHandling::FillZero).unwrap();
    assert_eq!(output, vec![0.0, 0.0, 0.0]);
}

#[test]
fn empty_array_is_noop() {
    let array = Float64Array::from(Vec::<f64>::new());
    let mut output = Vec::new();
    handle_float64_nulls(&mut output, &array, NullHandling::FillZero).unwrap();
    assert!(output.is_empty());
    handle_float64_nulls(&mut output, &array, NullHandling::Reject).unwrap();
    assert!(output.is_empty());
}
