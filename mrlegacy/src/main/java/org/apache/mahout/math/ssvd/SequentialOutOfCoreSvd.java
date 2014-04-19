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

import org.apache.mahout.math.CholeskyDecomposition;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.RandomTrinaryMatrix;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Sequential block-oriented out of core SVD algorithm.
 * <p/>
 * The basic algorithm (in-core version) is that we do a random projects, get a basis of that and
 * then re-project the original matrix using that basis.  This re-projected matrix allows us to get
 * an approximate SVD of the original matrix.
 * <p/>
 * The input to this program is a list of files that contain the sub-matrices A_i.  The result is a
 * vector of singular values and optionally files that contain the left and right singular vectors.
 * <p/>
 * Mathematically, to decompose A, we do this:
 * <p/>
 * Y = A * \Omega
 * <p/>
 * Q R = Y
 * <p/>
 * B = Q" A
 * <p/>
 * U D V' = B
 * <p/>
 * (Q U) D V' \approx A
 * <p/>
 * To do this out of core, we break A into blocks each with the same number of rows.  This gives a
 * block-wise version of Y.  As we are computing Y, we can also accumulate Y' Y and when done, we
 * can use a Cholesky decomposition to do the QR decomposition of Y in a latent form.  That gives us
 * B in block-wise form and we can do the same trick to get an LQ of B.  The L part can be
 * decomposed in memory.  Then we can recombine to get the final decomposition.
 * <p/>
 * The details go like this.  Start with a block form of A.
 * <p/>
 * Y_i = A_i * \Omega
 * <p/>
 * Instead of doing a QR decomposition of Y, we do a Cholesky decomposition of Y' Y.  This is a
 * small in-memory operation.  Q is large and dense and won't fit in memory.
 * <p/>
 * R' R = \sum_i Y_i' Y_i
 * <p/>
 * For reference, R is all we need to compute explicitly.  Q will be computed on the fly when
 * needed.
 * <p/>
 * Q = Y R^-1
 * <p/>
 * B = Q" A = \sum_i (A \Omega R^-1)' A_i
 * <p/>
 * As B is generated, it needs to be segmented in row-wise blocks since it is wide but not tall.
 * This storage requires something like a map-reduce to accumulate the partial sums.  In this code,
 * we do this by re-reading previously computed chunks and augmenting them.
 * <p/>
 * While the pieces of B are being computed, we can accumulate B B' in preparation for a second
 * Cholesky decomposition
 * <p/>
 * L L' = B B' = sum B_j B_j'
 * <p/>
 * Again, this is an LQ decomposition of BB', but we don't compute the Q part explicitly.  L will be
 * small and thus tractable.
 * <p/>
 * Finally, we do the actual SVD decomposition.
 * <p/>
 * U_0 D V_0' = L
 * <p/>
 * D contains the singular values of A.  The left and right singular values can be reconstructed
 * using Y and B.  Note that both of these reconstructions can be done with single passes through
 * the blocked forms of Y and B.
 * <p/>
 * U = A \Omega R^{-1} U_0
 * <p/>
 * V = B' L'^{-1} V_0
 */
public class SequentialOutOfCoreSvd {

  private final CholeskyDecomposition l2;
  private final SingularValueDecomposition svd;
  private final CholeskyDecomposition r2;
  private final int columnsPerSlice;
  private final int seed;
  private final int dim;

  public SequentialOutOfCoreSvd(Iterable<File> partsOfA, File tmpDir, int internalDimension, int columnsPerSlice)
    throws IOException {
    this.columnsPerSlice = columnsPerSlice;
    this.dim = internalDimension;

    seed = 1;
    Matrix y2 = null;

    // step 1, compute R as in R'R = Y'Y where Y = A \Omega
    for (File file : partsOfA) {
      MatrixWritable m = new MatrixWritable();
      DataInputStream in = new DataInputStream(new FileInputStream(file));
      try {
        m.readFields(in);
      } finally {
        in.close();
      }

      Matrix aI = m.get();
      Matrix omega = new RandomTrinaryMatrix(seed, aI.columnSize(), internalDimension, false);
      Matrix y = aI.times(omega);

      if (y2 == null) {
        y2 = y.transpose().times(y);
      } else {
        y2.assign(y.transpose().times(y), Functions.PLUS);
      }
    }
    r2 = new CholeskyDecomposition(y2);

    // step 2, compute B
    int ncols = 0;
    for (File file : partsOfA) {
      MatrixWritable m = new MatrixWritable();
      DataInputStream in = new DataInputStream(new FileInputStream(file));
      try {
        m.readFields(in);
      } finally {
        in.close();
      }
      Matrix aI = m.get();
      ncols = Math.max(ncols, aI.columnSize());

      Matrix omega = new RandomTrinaryMatrix(seed, aI.numCols(), internalDimension, false);
      for (int j = 0; j < aI.numCols(); j += columnsPerSlice) {
        Matrix yI = aI.times(omega);
        Matrix aIJ = aI.viewPart(0, aI.rowSize(), j, Math.min(columnsPerSlice, aI.columnSize() - j));
        Matrix bIJ = r2.solveRight(yI).transpose().times(aIJ);
        addToSavedCopy(bFile(tmpDir, j), bIJ);
      }
    }

    // step 3, compute BB', L and SVD(L)
    Matrix b2 = new DenseMatrix(internalDimension, internalDimension);
    MatrixWritable bTmp = new MatrixWritable();
    for (int j = 0; j < ncols; j += columnsPerSlice) {
      if (bFile(tmpDir, j).exists()) {
        DataInputStream in = new DataInputStream(new FileInputStream(bFile(tmpDir, j)));
        try {
          bTmp.readFields(in);
        } finally {
          in.close();
        }

        b2.assign(bTmp.get().times(bTmp.get().transpose()), Functions.PLUS);
      }
    }
    l2 = new CholeskyDecomposition(b2);
    svd = new SingularValueDecomposition(l2.getL());
  }

  public void computeV(File tmpDir, int ncols) throws IOException {
    // step 5, compute pieces of V
    for (int j = 0; j < ncols; j += columnsPerSlice) {
      File bPath = bFile(tmpDir, j);
      if (bPath.exists()) {
        MatrixWritable m = new MatrixWritable();
        DataInputStream in = new DataInputStream(new FileInputStream(bPath));
        try {
          m.readFields(in);
        } finally {
          in.close();
        }
        m.set(l2.solveRight(m.get().transpose()).times(svd.getV()));
        DataOutputStream out = new DataOutputStream(new FileOutputStream(
            new File(tmpDir, String.format("V-%s", bPath.getName().replaceAll(".*-", "")))));
        try {
          m.write(out);
        } finally {
          out.close();
        }
      }
    }
  }

  public void computeU(Iterable<File> partsOfA, File tmpDir) throws IOException {
    // step 4, compute pieces of U
    for (File file : partsOfA) {
      MatrixWritable m = new MatrixWritable();
      m.readFields(new DataInputStream(new FileInputStream(file)));
      Matrix aI = m.get();

      Matrix y = aI.times(new RandomTrinaryMatrix(seed, aI.numCols(), dim, false));
      Matrix uI = r2.solveRight(y).times(svd.getU());
      m.set(uI);
      DataOutputStream out = new DataOutputStream(new FileOutputStream(
          new File(tmpDir, String.format("U-%s", file.getName().replaceAll(".*-", "")))));
      try {
        m.write(out);
      } finally {
        out.close();
      }
    }
  }

  private static void addToSavedCopy(File file, Matrix matrix) throws IOException {
    MatrixWritable mw = new MatrixWritable();
    if (file.exists()) {
      DataInputStream in = new DataInputStream(new FileInputStream(file));
      try {
        mw.readFields(in);
      } finally {
        in.close();
      }
      mw.get().assign(matrix, Functions.PLUS);
    } else {
      mw.set(matrix);
    }
    DataOutputStream out = new DataOutputStream(new FileOutputStream(file));
    try {
      mw.write(out);
    } finally {
      out.close();
    }
  }

  private static File bFile(File tmpDir, int j) {
    return new File(tmpDir, String.format("B-%09d", j));
  }

  public Vector getSingularValues() {
    return new DenseVector(svd.getSingularValues());
  }
}
