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

import org.apache.mahout.benchmark.BenchmarkRunner.BenchmarkFn;
import org.apache.mahout.math.Vector;

public class PlusBenchmark {

  private static final String PLUS = "Plus";
  private final VectorBenchmarks mark;

  public PlusBenchmark(VectorBenchmarks mark) {
    this.mark = mark;
  }

  public void benchmark() {
    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        Vector v = mark.vectors[0][mark.vIndex(i)].plus(mark.vectors[0][mark.vIndex(randIndex())]);
        return depends(v);
      }
    }), PLUS, DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        Vector v = mark.vectors[1][mark.vIndex(i)].plus(mark.vectors[1][mark.vIndex(randIndex())]);
        return depends(v);
      }
    }), PLUS, RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        Vector v = mark.vectors[2][mark.vIndex(i)].plus(mark.vectors[2][mark.vIndex(randIndex())]);
        return depends(v);
      }
    }), PLUS, SEQ_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        Vector v = mark.vectors[0][mark.vIndex(i)].plus(mark.vectors[1][mark.vIndex(randIndex())]);
        return depends(v);
      }
    }), PLUS, DENSE_FN_RAND);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        Vector v = mark.vectors[0][mark.vIndex(i)].plus(mark.vectors[2][mark.vIndex(randIndex())]);
        return depends(v);
      }
    }), PLUS, DENSE_FN_SEQ);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        Vector v = mark.vectors[1][mark.vIndex(i)].plus(mark.vectors[0][mark.vIndex(randIndex())]);
        return depends(v);
      }
    }), PLUS, RAND_FN_DENSE);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        Vector v = mark.vectors[1][mark.vIndex(i)].plus(mark.vectors[2][mark.vIndex(randIndex())]);
        return depends(v);
      }
    }), PLUS, RAND_FN_SEQ);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        Vector v = mark.vectors[2][mark.vIndex(i)].plus(mark.vectors[0][mark.vIndex(randIndex())]);
        return depends(v);
      }
    }), PLUS, SEQ_FN_DENSE);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        Vector v = mark.vectors[2][mark.vIndex(i)].plus(mark.vectors[1][mark.vIndex(randIndex())]);
        return depends(v);
      }
    }), PLUS, SEQ_FN_RAND);
  }
}
