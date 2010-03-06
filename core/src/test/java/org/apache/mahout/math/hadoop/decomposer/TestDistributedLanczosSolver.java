package org.apache.mahout.math.hadoop.decomposer;

import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.canopy.TestCanopyCreation;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.decomposer.SolverTest;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.TestDistributedRowMatrix;

import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class TestDistributedLanczosSolver extends SolverTest {

  public TestDistributedLanczosSolver(String name) {
    super(name);
  }

  public void doTestDistributedLanczosSolver(boolean symmetric) throws Exception {
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    DistributedRowMatrix corpus = TestDistributedRowMatrix.randomDistributedMatrix(500,
        450, 400, 10, 10.0, symmetric, "testdata");
    corpus.configure(new JobConf());
    DistributedLanczosSolver solver = new DistributedLanczosSolver();
    int desiredRank = 30;
    Matrix eigenVectors = new DenseMatrix(desiredRank, corpus.numCols());
    List<Double> eigenValues = new ArrayList<Double>();
    solver.solve(corpus, desiredRank, eigenVectors, eigenValues, symmetric);
    assertOrthonormal(eigenVectors);
    assertEigen(eigenVectors, corpus, eigenVectors.numRows() / 2, 0.01, symmetric);
  }

  public void testDistributedLanczosSolver() throws Exception {
  //  doTestDistributedLanczosSolver(false);
  //  TestCanopyCreation.rmr("testData");
    doTestDistributedLanczosSolver(true);
  }

  @Override
  public void tearDown() throws Exception {
    TestCanopyCreation.rmr("testData");
    super.tearDown();
  }


}
