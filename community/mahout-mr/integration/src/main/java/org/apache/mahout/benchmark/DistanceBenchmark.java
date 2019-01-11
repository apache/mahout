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

import static org.apache.mahout.benchmark.VectorBenchmarks.DENSE_FN_RAND;
import static org.apache.mahout.benchmark.VectorBenchmarks.DENSE_FN_SEQ;
import static org.apache.mahout.benchmark.VectorBenchmarks.DENSE_VECTOR;
import static org.apache.mahout.benchmark.VectorBenchmarks.RAND_FN_DENSE;
import static org.apache.mahout.benchmark.VectorBenchmarks.RAND_FN_SEQ;
import static org.apache.mahout.benchmark.VectorBenchmarks.RAND_SPARSE_VECTOR;
import static org.apache.mahout.benchmark.VectorBenchmarks.SEQ_FN_DENSE;
import static org.apache.mahout.benchmark.VectorBenchmarks.SEQ_FN_RAND;
import static org.apache.mahout.benchmark.VectorBenchmarks.SEQ_SPARSE_VECTOR;

import org.apache.mahout.benchmark.BenchmarkRunner.BenchmarkFnD;
import org.apache.mahout.common.distance.DistanceMeasure;

public class DistanceBenchmark {
  private final VectorBenchmarks mark;

  public DistanceBenchmark(VectorBenchmarks mark) {
    this.mark = mark;
  }

  public void benchmark(final DistanceMeasure measure) {
    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return measure.distance(mark.vectors[0][mark.vIndex(i)], mark.vectors[0][mark.vIndex(randIndex())]);
      }
    }), measure.getClass().getName(), DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return measure.distance(mark.vectors[1][mark.vIndex(i)], mark.vectors[1][mark.vIndex(randIndex())]);
      }
    }), measure.getClass().getName(), RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return measure.distance(mark.vectors[2][mark.vIndex(i)], mark.vectors[2][mark.vIndex(randIndex())]);
      }
    }), measure.getClass().getName(), SEQ_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return measure.distance(mark.vectors[0][mark.vIndex(i)], mark.vectors[1][mark.vIndex(randIndex())]);
      }
    }), measure.getClass().getName(), DENSE_FN_RAND);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return measure.distance(mark.vectors[0][mark.vIndex(i)], mark.vectors[2][mark.vIndex(randIndex())]);
      }
    }), measure.getClass().getName(), DENSE_FN_SEQ);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return measure.distance(mark.vectors[1][mark.vIndex(i)], mark.vectors[0][mark.vIndex(randIndex())]);
      }
    }), measure.getClass().getName(), RAND_FN_DENSE);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return measure.distance(mark.vectors[1][mark.vIndex(i)], mark.vectors[2][mark.vIndex(randIndex())]);
      }
    }), measure.getClass().getName(), RAND_FN_SEQ);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return measure.distance(mark.vectors[2][mark.vIndex(i)], mark.vectors[0][mark.vIndex(randIndex())]);
      }
    }), measure.getClass().getName(), SEQ_FN_DENSE);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return measure.distance(mark.vectors[2][mark.vIndex(i)], mark.vectors[1][mark.vIndex(randIndex())]);
      }
    }), measure.getClass().getName(), SEQ_FN_RAND);
  }
}
