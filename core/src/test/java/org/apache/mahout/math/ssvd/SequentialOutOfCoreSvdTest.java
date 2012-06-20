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

import com.google.common.collect.Lists;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.RandomTrinaryMatrix;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
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
    tmpDir = getTestTempDir("matrix");
  }

  @Test
  public void testSingularValues() throws IOException {
    Matrix A = lowRankMatrix(tmpDir, "A", 200, 970, 1020);

    List<File> partsOfA = Arrays.asList(tmpDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String fileName) {
        return fileName.matches("A-.*");
      }
    }));

    // rearrange A to make sure we don't depend on lexical ordering.
    partsOfA = Lists.reverse(partsOfA);
    SequentialOutOfCoreSvd s = new SequentialOutOfCoreSvd(partsOfA, tmpDir, 100, 210);
    SequentialBigSvd svd = new SequentialBigSvd(A, 100);

    Vector reference = new DenseVector(svd.getSingularValues()).viewPart(0, 6);
    Vector actual = s.getSingularValues().viewPart(0, 6);
    assertEquals(0, reference.minus(actual).maxValue(), 1.0e-9);

    s.computeU(partsOfA, tmpDir);
    Matrix u = readBlockMatrix(Arrays.asList(tmpDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String fileName) {
        return fileName.matches("U-.*");
      }
    })));

    s.computeV(tmpDir, A.columnSize());
    Matrix v = readBlockMatrix(Arrays.asList(tmpDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String fileName) {
        return fileName.matches("V-.*");
      }
    })));

    // The values in A are pretty big so this is a pretty tight relative tolerance
    assertEquals(0, A.minus(u.times(new DiagonalMatrix(s.getSingularValues())).times(v.transpose())).aggregate(Functions.PLUS, Functions.ABS), 1.0e-7);
  }

  /**
   * Reads a list of files that contain a column of blocks.  It is assumed that the files
   * can be sorted lexicographically to determine the order they should be stacked.  It
   * is also assumed here that all blocks will be the same size except the last one which
   * may be shorter than the others.
   * @param files  The list of files to read.
   * @return  The row-wise concatenation of the matrices in the files.
   * @throws IOException If we can't read the sub-matrices.
   */
  private static Matrix readBlockMatrix(List<File> files) throws IOException {
    // force correct ordering
    Collections.sort(files);

    // initially, we don't know what size buffer to hold
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
        // now we can set an upper bound on how large our result will be
        nrows = m.get().rowSize() * files.size();
        ncols = m.get().columnSize();
        r = new DenseMatrix(nrows, ncols);
      }
      r.viewPart(row, m.get().rowSize(), 0, r.columnSize()).assign(m.get());
      row += m.get().rowSize();
    }
    // at the end, row will have the true size of the result
    if (row != nrows && r != null) {
      // and if that isn't the size of the buffer, we need to crop the result a bit
      r = r.viewPart(0, row, 0, ncols);
    }
    return r;
  }

  @Test
  public void testLeftVectors() throws IOException {
    Matrix A = lowRankMatrixInMemory(20, 20);

    SequentialBigSvd s = new SequentialBigSvd(A, 6);
    SingularValueDecomposition svd = new SingularValueDecomposition(A);

    // can only check first few singular vectors
    Matrix u1 = svd.getU().viewPart(0, 20, 0, 3).assign(Functions.ABS);
    Matrix u2 = s.getU().viewPart(0, 20, 0, 3).assign(Functions.ABS);
    assertEquals(u1, u2);
  }

  private static Matrix lowRankMatrixInMemory(int rows, int columns) throws IOException {
    return lowRankMatrix(null, null, 0, rows, columns);
  }

  private static void assertEquals(Matrix u1, Matrix u2) {
    assertEquals(0.0, u1.minus(u2).aggregate(Functions.MAX, Functions.ABS), 1.0e-10);
  }

  @Test
  public void testRightVectors() throws IOException {
    Matrix A = lowRankMatrixInMemory(20, 20);

    SequentialBigSvd s = new SequentialBigSvd(A, 6);
    SingularValueDecomposition svd = new SingularValueDecomposition(A);

    Matrix v1 = svd.getV().viewPart(0, 20, 0, 3).assign(Functions.ABS);
    Matrix v2 = s.getV().viewPart(0, 20, 0, 3).assign(Functions.ABS);
    assertEquals(v1, v2);
  }

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

    if (tmpDir != null) {
      for (int i = 0; i < a.rowSize(); i += rowsPerSlice) {
        MatrixWritable m = new MatrixWritable(a.viewPart(i, Math.min(a.rowSize() - i, rowsPerSlice), 0, a.columnSize()));
        DataOutputStream out = new DataOutputStream(new FileOutputStream(new File(tmpDir, String.format("%s-%09d", aBase, i))));
        try {
          m.write(out);
        } finally {
          out.close();
        }
      }
    }
    return a;
  }
}
