package org.apache.mahout.math.ssvd;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.RandomTrinaryMatrix;
import org.apache.mahout.math.Vector;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class SequentialOutOfCoreSvdTest {

  private File tmpDir;

  @Before
  public void setup() throws IOException {
    tmpDir = File.createTempFile("matrix", "");
    Assert.assertTrue(tmpDir.delete());
    Assert.assertTrue(tmpDir.mkdir());
  }

  @After
  public void tearDown() {
    for (File f : tmpDir.listFiles()) {
      Assert.assertTrue(f.delete());
    }
    Assert.assertTrue(tmpDir.delete());
  }

  @Test
  public void testSingularValues() throws IOException {
    Matrix A = lowRankMatrix(tmpDir, "A", 200);

    List<File> partsOfA = Arrays.asList(tmpDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String s) {
        return s.matches("A-.*");
      }
    }));
    SequentialOutOfCoreSvd s = new SequentialOutOfCoreSvd(partsOfA, "U", "V", tmpDir, 100, 200);
    SequentialBigSvd svd = new SequentialBigSvd(A, 20);

    Vector reference = new DenseVector(svd.getSingularValues()).viewPart(0, 6);
    assertEquals(0, reference.minus(s.getSingularValues().viewPart(0, 6)).maxValue(), 1e-9);

    s.computeU(partsOfA, "U-", tmpDir);
    Matrix u = readBlockMatrix(Arrays.asList(tmpDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String s) {
        return s.matches("U-.*");
      }
    })), A.rowSize(), 15);

    s.computeV(tmpDir, "V-", A.columnSize());
    Matrix v = readBlockMatrix(Arrays.asList(tmpDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String s) {
        return s.matches("V-.*");
      }
    })), A.rowSize(), 15);

    assertEquals(A, u.times(new DiagonalMatrix(s.getSingularValues())).times(v.transpose()));
  }

  private Matrix readBlockMatrix(List<File> files, int nrows, int ncols) throws IOException {
    Collections.sort(files);
    Matrix r = new DenseMatrix(nrows, ncols);

    MatrixWritable m = new MatrixWritable();

    int row = 0;
    for (File file : files) {
      DataInputStream in = new DataInputStream(new FileInputStream(file));
      m.readFields(in);
      in.close();
      r.viewPart(row, m.get().rowSize(), 0, r.columnSize()).assign(m.get());
      row += m.get().rowSize();
    }
    return r;
  }

//  @Test
//  public void testLeftVectors() {
//    Matrix A = lowRankMatrix();
//
//    SequentialBigSvd s = new SequentialBigSvd(A, 6);
//    SingularValueDecomposition svd = new SingularValueDecomposition(A);
//
//    // can only check first few singular vectors
//    Matrix u1 = svd.getU().viewPart(0, 20, 0, 3).assign(Functions.ABS);
//    Matrix u2 = s.getU().viewPart(0, 20, 0, 3).assign(Functions.ABS);
//    assertEquals(u1, u2);
//  }
//
//  private void assertEquals(Matrix u1, Matrix u2) {
//    Assert.assertEquals(0.0, u1.minus(u2).aggregate(Functions.MAX, Functions.ABS), 1e-10);
//  }
//
//  private void assertEquals(Vector u1, Vector u2) {
//    Assert.assertEquals(0.0, u1.minus(u2).aggregate(Functions.MAX, Functions.ABS), 1e-10);
//  }
//
//  @Test
//  public void testRightVectors() {
//    Matrix A = lowRankMatrix();
//
//    SequentialBigSvd s = new SequentialBigSvd(A, 6);
//    SingularValueDecomposition svd = new SingularValueDecomposition(A);
//
//    Matrix v1 = svd.getV().viewPart(0, 20, 0, 3).assign(Functions.ABS);
//    Matrix v2 = s.getV().viewPart(0, 20, 0, 3).assign(Functions.ABS);
//    assertEquals(v1, v2);
//  }

  private Matrix lowRankMatrix(File tmpDir, String aBase, int rowsPerSlice) throws IOException {
    int rank = 10;
    Matrix u = new RandomTrinaryMatrix(1, 1000, rank, false);
    Matrix d = new DenseMatrix(rank, rank);
    d.set(0, 0, 5);
    d.set(1, 1, 3);
    d.set(2, 2, 1);
    d.set(3, 3, 0);
    Matrix v = new RandomTrinaryMatrix(2, 1000, rank, false);
    Matrix a = u.times(d).times(v.transpose());

    for (int i = 0; i < a.rowSize(); i += rowsPerSlice) {
      MatrixWritable m = new MatrixWritable(a.viewPart(i, rowsPerSlice, 0, a.columnSize()));
      DataOutputStream out = new DataOutputStream(new FileOutputStream(new File(tmpDir, String.format("%s-%09d", aBase, i))));
      try {
        m.write(out);
      } finally {
        out.close();
      }
    }
    return a;
  }

}
