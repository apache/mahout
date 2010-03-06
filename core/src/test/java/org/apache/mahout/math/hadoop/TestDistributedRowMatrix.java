package org.apache.mahout.math.hadoop;

import junit.framework.TestCase;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.canopy.TestCanopyCreation;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.decomposer.SolverTest;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class TestDistributedRowMatrix extends TestCase {

  private static final String TESTDATA = "testdata";

  public TestDistributedRowMatrix() {
    super();
  }

  @Override
  public void setUp() throws Exception {
    File testData = new File(TESTDATA);
    if (testData.exists()) {
      TestCanopyCreation.rmr(TESTDATA);
    }
    testData.mkdir();
  }

  @Override
  public void tearDown() throws Exception {
    TestCanopyCreation.rmr(TESTDATA);
    super.tearDown();
  }

  public static void assertEquals(double d1, double d2, double errorTolerance) {
    assertTrue(Math.abs(d1-d2) < errorTolerance);
  }

  public static void assertEquals(VectorIterable m, VectorIterable mtt, double errorTolerance) {
    Iterator<MatrixSlice> mIt = m.iterateAll();
    Iterator<MatrixSlice> mttIt = mtt.iterateAll();
    Map<Integer, Vector> mMap = new HashMap<Integer,Vector>();
    Map<Integer, Vector> mttMap = new HashMap<Integer, Vector>();
    while(mIt.hasNext() && mttIt.hasNext()) {
      MatrixSlice ms = mIt.next();
      mMap.put(ms.index(), ms.vector());
      MatrixSlice mtts = mttIt.next();
      mttMap.put(mtts.index(), mtts.vector());
    }
    for(Integer i : mMap.keySet()) {
      if(mMap.get(i) == null || mttMap.get(i) == null) {
        assertTrue(mMap.get(i) == null || mMap.get(i).norm(2) == 0);
        assertTrue(mttMap.get(i) == null || mttMap.get(i).norm(2) == 0);
      } else {
        assertTrue(mMap.get(i).getDistanceSquared(mttMap.get(i)) < errorTolerance);
      }
    }
  }

  public void testTranspose() throws Exception {
    DistributedRowMatrix m = randomDistributedMatrix(10, 9, 5, 4, 1.0, false);
    DistributedRowMatrix mt = m.transpose();
    mt.setOutputTempPathString(new Path(m.getOutputTempPath().getParent(), "/tmpOutTranspose").toString());
    DistributedRowMatrix mtt = mt.transpose();
    assertEquals(m, mtt, 1e-9);
  }

  public void testMatrixTimesVector() throws Exception {
    Vector v = new RandomAccessSparseVector(50);
    v.assign(1.0);
    Matrix m = SolverTest.randomSequentialAccessSparseMatrix(100, 90, 50, 20, 1.0);
    DistributedRowMatrix dm = randomDistributedMatrix(100, 90, 50, 20, 1.0, false);

    Vector expected = m.times(v);
    Vector actual = dm.times(v);
    assertEquals(expected.getDistanceSquared(actual), 0.0, 1e-9);
  }

  public void testMatrixTimesSquaredVector() throws Exception {
    Vector v = new RandomAccessSparseVector(50);
    v.assign(1.0);
    Matrix m = SolverTest.randomSequentialAccessSparseMatrix(100, 90, 50, 20, 1.0);
    DistributedRowMatrix dm = randomDistributedMatrix(100, 90, 50, 20, 1.0, false);

    Vector expected = m.timesSquared(v);
    Vector actual = dm.timesSquared(v);
    assertEquals(expected.getDistanceSquared(actual), 0.0, 1e-9);
  }

  public void testMatrixTimesMatrix() throws Exception {
    Matrix inputA = SolverTest.randomSequentialAccessSparseMatrix(20, 19, 15, 5, 10.0);
    Matrix inputB = SolverTest.randomSequentialAccessSparseMatrix(20, 13, 25, 10, 5.0);
    Matrix expected = inputA.transpose().times(inputB);

    DistributedRowMatrix distA = randomDistributedMatrix(20, 19, 15, 5, 10.0, false, "/distA");
    DistributedRowMatrix distB = randomDistributedMatrix(20, 13, 25, 10, 5.0, false, "/distB");
    DistributedRowMatrix product = distA.times(distB);

    assertEquals(expected, product, 1e-9);
  }

  public static DistributedRowMatrix randomDistributedMatrix(int numRows,
                                                             int nonNullRows,
                                                             int numCols,
                                                             int entriesPerRow,
                                                             double entryMean,
                                                             boolean isSymmetric) throws Exception {
    return randomDistributedMatrix(numRows, nonNullRows, numCols, entriesPerRow, entryMean, isSymmetric, "");
  }

  public static DistributedRowMatrix randomDistributedMatrix(int numRows,
                                                             int nonNullRows,
                                                             int numCols,
                                                             int entriesPerRow,
                                                             double entryMean,
                                                             boolean isSymmetric,
                                                             String baseTmpDir) throws IOException {
    baseTmpDir = TESTDATA + baseTmpDir;
    Matrix c = SolverTest.randomSequentialAccessSparseMatrix(numRows, nonNullRows, numCols, entriesPerRow, entryMean);
    if(isSymmetric) {
      c = c.times(c.transpose());
    }
    final Matrix m = c;
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
    }, true, baseTmpDir + "/distMatrix/part-00000", fs, conf);

    DistributedRowMatrix distMatrix = new DistributedRowMatrix(baseTmpDir + "/distMatrix",
                                                               baseTmpDir + "/tmpOut",
                                                               m.numRows(),
                                                               m.numCols());
    distMatrix.configure(new JobConf(conf));

    return distMatrix;
  }
}
