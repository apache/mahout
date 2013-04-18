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
