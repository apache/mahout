package org.apache.mahout.math.hadoop.decomposer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.canopy.TestCanopyCreation;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.decomposer.SolverTest;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


public class TestDistributedLanczosSolver extends SolverTest {

  public TestDistributedLanczosSolver(String name) {
    super(name);
  }

  public void testDistributedLanczosSolver() throws Exception {
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    DistributedRowMatrix corpus = randomDistributedMatrix(1000, 900, 800, 100, 10.0, "testdata");
    corpus.configure(new JobConf());
    DistributedLanczosSolver solver = new DistributedLanczosSolver();
    int desiredRank = 30;
    Matrix eigenVectors = new DenseMatrix(desiredRank, 800);
    List<Double> eigenValues = new ArrayList<Double>();
    solver.solve(corpus, desiredRank, eigenVectors, eigenValues);
    assertOrthonormal(eigenVectors);
    assertEigen(eigenVectors, corpus, eigenVectors.numRows() / 2, 0.01);
  }

  @Override
  public void tearDown() throws Exception {
    TestCanopyCreation.rmr("testData");
    super.tearDown();
  }


  public static DistributedRowMatrix randomDistributedMatrix(int numRows,
                                                          int nonNullRows,
                                                          int numCols,
                                                          int entriesPerRow,
                                                          double entryMean,
                                                          String baseTmpDir) throws IOException {
    final Matrix m = randomSequentialAccessSparseMatrix(numRows, nonNullRows, numCols, entriesPerRow, entryMean);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    ClusteringTestUtils.writePointsToFile(new Iterable<VectorWritable>() {
      @Override
      public Iterator<VectorWritable> iterator() {
        final Iterator<MatrixSlice> it = m.iterator();
        final VectorWritable v = new VectorWritable();
        return new Iterator<VectorWritable>() {
          @Override
          public boolean hasNext() { return it.hasNext(); }
          @Override
          public VectorWritable next() {
            MatrixSlice slice = it.next();
            v.set(slice.vector());
            return v;
          }
          @Override
          public void remove() { it.remove(); }
        };
      }
    }, baseTmpDir + "/distMatrix", fs, conf);

    DistributedRowMatrix distMatrix = new DistributedRowMatrix(baseTmpDir + "/distMatrix",
                                                               baseTmpDir + "/tmpOut",
                                                               m.numRows(),
                                                               m.numCols());

    return distMatrix;
  }
}
