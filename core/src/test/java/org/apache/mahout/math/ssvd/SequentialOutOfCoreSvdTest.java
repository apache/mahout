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

package org.apache.mahout.math.ssvd;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.RandomTrinaryMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.junit.After;
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

public final class SequentialOutOfCoreSvdTest extends MahoutTestCase {

  private File tmpDir;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    tmpDir = File.createTempFile("matrix", "");
    assertTrue(tmpDir.delete());
    assertTrue(tmpDir.mkdir());
  }

  @Override
  @After
  public void tearDown() throws Exception {
    for (File f : tmpDir.listFiles()) {
      assertTrue(f.delete());
    }
    assertTrue(tmpDir.delete());
    super.tearDown();
  }

  @Test
  public void testSingularValues() throws IOException {
    Matrix A = lowRankMatrix(tmpDir, "A", 200, 970, 1020);

    List<File> partsOfA = Arrays.asList(tmpDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String s) {
        return s.matches("A-.*");
      }
    }));
    SequentialOutOfCoreSvd s = new SequentialOutOfCoreSvd(partsOfA, "U", "V", tmpDir, 100, 210);
    SequentialBigSvd svd = new SequentialBigSvd(A, 100);

    Vector reference = new DenseVector(svd.getSingularValues()).viewPart(0, 6);
    Vector actual = s.getSingularValues().viewPart(0, 6);
    assertEquals(0, reference.minus(actual).maxValue(), 1e-9);

    s.computeU(partsOfA, "U-", tmpDir);
    Matrix u = readBlockMatrix(Arrays.asList(tmpDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String s) {
        return s.matches("U-.*");
      }
    })));

    s.computeV(tmpDir, "V-", A.columnSize());
    Matrix v = readBlockMatrix(Arrays.asList(tmpDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String s) {
        return s.matches("V-.*");
      }
    })));

    // The values in A are pretty big so this is a pretty tight relative tolerance
    assertEquals(0, A.minus(u.times(new DiagonalMatrix(s.getSingularValues())).times(v.transpose())).aggregate(Functions.PLUS, Functions.ABS), 1e-7);
  }

  private static Matrix readBlockMatrix(List<File> files) throws IOException {
    Collections.sort(files);
    int nrows = -1;
    int ncols = -1;
    Matrix r = null;

    MatrixWritable m = new MatrixWritable();

    int row = 0;
    for (File file : files) {
      DataInputStream in = new DataInputStream(new FileInputStream(file));
      m.readFields(in);
      in.close();
      if (nrows == -1) {
        nrows = m.get().rowSize() * files.size();
        ncols = m.get().columnSize();
        r = new DenseMatrix(nrows, ncols);
      }
      r.viewPart(row, m.get().rowSize(), 0, r.columnSize()).assign(m.get());
      row += m.get().rowSize();
    }
    if (row != nrows && r != null) {
      r = r.viewPart(0, row, 0, ncols);
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
//    assertEquals(0.0, u1.minus(u2).aggregate(Functions.MAX, Functions.ABS), 1e-10);
//  }
//
//  private void assertEquals(Vector u1, Vector u2) {
//    assertEquals(0.0, u1.minus(u2).aggregate(Functions.MAX, Functions.ABS), 1e-10);
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

  private static Matrix lowRankMatrix(File tmpDir, String aBase, int rowsPerSlice, int rows, int columns) throws IOException {
    int rank = 10;
    Matrix u = new RandomTrinaryMatrix(1, rows, rank, false);
    Matrix d = new DenseMatrix(rank, rank);
    d.set(0, 0, 5);
    d.set(1, 1, 3);
    d.set(2, 2, 1);
    d.set(3, 3, 0.5);
    Matrix v = new RandomTrinaryMatrix(2, columns, rank, false);
    Matrix a = u.times(d).times(v.transpose());

    for (int i = 0; i < a.rowSize(); i += rowsPerSlice) {
      MatrixWritable m = new MatrixWritable(a.viewPart(i, Math.min(a.rowSize() - i, rowsPerSlice), 0, a.columnSize()));
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
