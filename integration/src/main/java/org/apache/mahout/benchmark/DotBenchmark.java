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
import org.apache.mahout.benchmark.BenchmarkRunner.BenchmarkFnD;

public class DotBenchmark {
  private static final String DOT_PRODUCT = "DotProduct";
  private static final String NORM1 = "Norm1";
  private static final String NORM2 = "Norm2";
  private static final String LOG_NORMALIZE = "LogNormalize";
  private final VectorBenchmarks mark;

  public DotBenchmark(VectorBenchmarks mark) {
    this.mark = mark;
  }

  public void benchmark() {
    benchmarkDot();
    benchmarkNorm1();
    benchmarkNorm2();
    benchmarkLogNormalize();
  }

  private void benchmarkLogNormalize() {
    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        return depends(mark.vectors[0][mark.vIndex(i)].logNormalize());
      }
    }), LOG_NORMALIZE, DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        return depends(mark.vectors[1][mark.vIndex(i)].logNormalize());
      }
    }), LOG_NORMALIZE, RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        return depends(mark.vectors[2][mark.vIndex(i)].logNormalize());
      }
    }), LOG_NORMALIZE, SEQ_SPARSE_VECTOR);
  }

  private void benchmarkNorm1() {
    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[0][mark.vIndex(i)].norm(1);
      }
    }), NORM1, DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[1][mark.vIndex(i)].norm(1);
      }
    }), NORM1, RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[2][mark.vIndex(i)].norm(1);
      }
    }), NORM1, SEQ_SPARSE_VECTOR);
  }

  private void benchmarkNorm2() {
    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[0][mark.vIndex(i)].norm(2);
      }
    }), NORM2, DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[1][mark.vIndex(i)].norm(2);
      }
    }), NORM2, RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[2][mark.vIndex(i)].norm(2);
      }
    }), NORM2, SEQ_SPARSE_VECTOR);
  }

  private void benchmarkDot() {
    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[0][mark.vIndex(i)].dot(mark.vectors[0][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[1][mark.vIndex(i)].dot(mark.vectors[1][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[2][mark.vIndex(i)].dot(mark.vectors[2][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, SEQ_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[0][mark.vIndex(i)].dot(mark.vectors[1][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, DENSE_FN_RAND);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[0][mark.vIndex(i)].dot(mark.vectors[2][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, DENSE_FN_SEQ);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[1][mark.vIndex(i)].dot(mark.vectors[0][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, RAND_FN_DENSE);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[1][mark.vIndex(i)].dot(mark.vectors[2][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, RAND_FN_SEQ);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[2][mark.vIndex(i)].dot(mark.vectors[0][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, SEQ_FN_DENSE);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[2][mark.vIndex(i)].dot(mark.vectors[1][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, SEQ_FN_RAND);
  }

  public static void main(String[] args) {
    VectorBenchmarks mark = new VectorBenchmarks(1000000, 100, 1000, 10, 1);
    mark.createData();
    new DotBenchmark(mark).benchmarkNorm2();
    System.out.println(mark);
  }
}
