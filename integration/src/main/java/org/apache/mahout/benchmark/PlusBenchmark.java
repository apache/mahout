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
