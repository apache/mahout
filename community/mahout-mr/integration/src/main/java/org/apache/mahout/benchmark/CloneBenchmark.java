/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.benchmark;

import static org.apache.mahout.benchmark.VectorBenchmarks.DENSE_VECTOR;
import static org.apache.mahout.benchmark.VectorBenchmarks.RAND_SPARSE_VECTOR;
import static org.apache.mahout.benchmark.VectorBenchmarks.SEQ_SPARSE_VECTOR;

import org.apache.mahout.benchmark.BenchmarkRunner.BenchmarkFn;

public class CloneBenchmark {
  public static final String CLONE = "Clone";
  private final VectorBenchmarks mark;

  public CloneBenchmark(VectorBenchmarks mark) {
    this.mark = mark;
  }

  public void benchmark() {
    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        mark.vectors[0][mark.vIndex(i)] = mark.vectors[0][mark.vIndex(i)].clone();

        return depends(mark.vectors[0][mark.vIndex(i)]);
      }
    }), CLONE, DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        mark.vectors[1][mark.vIndex(i)] = mark.vectors[1][mark.vIndex(i)].clone();

        return depends(mark.vectors[1][mark.vIndex(i)]);
      }
    }), CLONE, RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        mark.vectors[2][mark.vIndex(i)] = mark.vectors[2][mark.vIndex(i)].clone();

        return depends(mark.vectors[2][mark.vIndex(i)]);
      }
    }), CLONE, SEQ_SPARSE_VECTOR);
  }
}
