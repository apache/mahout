package org.apache.mahout.math.hadoop.solver;

import java.io.File;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.TestDistributedRowMatrix;
import org.junit.Test;


public class TestDistributedConjugateGradientSolver extends MahoutTestCase
{
  private Vector randomVector(int size, double entryMean) {
    DenseVector v = new DenseVector(size);
    Random r = new Random(1234L);
    
    for (int i = 0; i < size; ++i) {
      v.setQuick(i, r.nextGaussian() * entryMean);
    }
    
    return v;
  }

  @Test
  public void testSolver() throws Exception {
    File testData = getTestTempDir("testdata");
    DistributedRowMatrix matrix = new TestDistributedRowMatrix().randomDistributedMatrix(
        10, 10, 10, 10, 10.0, true, testData.getAbsolutePath());
    matrix.setConf(new Configuration());
    Vector vector = randomVector(matrix.numCols(), 10.0);
    
    DistributedConjugateGradientSolver solver = new DistributedConjugateGradientSolver();
    Vector x = solver.solve(matrix, vector);

    Vector solvedVector = matrix.times(x);    
    double distance = Math.sqrt(vector.getDistanceSquared(solvedVector));
    assertEquals(0.0, distance, EPSILON);
  }
}
