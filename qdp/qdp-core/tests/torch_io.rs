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

#[cfg(feature = "pytorch")]
mod pytorch_tests {
    use qdp_core::io::read_torch_batch;
    use qdp_core::reader::DataReader;
    use qdp_core::readers::TorchReader;
    use std::fs;
    use tch::Tensor;

    #[test]
    fn test_torch_reader_basic_1d() {
        let temp_path = "/tmp/test_torch_basic_1d.pt";
        let sample_size = 12;
        let data: Vec<f64> = (0..sample_size).map(|i| i as f64).collect();

        let tensor = Tensor::from_slice(&data);
        tensor.save(temp_path).unwrap();

        let mut reader = TorchReader::new(temp_path).unwrap();
        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        assert_eq!(read_samples, 1);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data, data);

        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_read_torch_batch_function_2d() {
        let temp_path = "/tmp/test_torch_batch_fn.pt";
        let num_samples = 4;
        let sample_size = 3;
        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();

        let tensor = Tensor::from_slice(&data).reshape([num_samples as i64, sample_size as i64]);
        tensor.save(temp_path).unwrap();

        let (read_data, read_samples, read_size) = read_torch_batch(temp_path).unwrap();

        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data, data);

        fs::remove_file(temp_path).unwrap();
    }
}
