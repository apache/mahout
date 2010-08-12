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

/*
Copyright ? 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
is hereby granted without fee, provided that the above copyright notice appear in all copies and
that both that copyright notice and this permission notice appear in supporting documentation.
CERN makes no representations about the suitability of this software for any purpose.
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math;

import org.apache.mahout.math.function.Functions;


/**
 For an <tt>m x n</tt> matrix <tt>A</tt> with <tt>m >= n</tt>, the QR decomposition is an <tt>m x n</tt>
 orthogonal matrix <tt>Q</tt> and an <tt>n x n</tt> upper triangular matrix <tt>R</tt> so that
 <tt>A = Q*R</tt>.
 <P>
 The QR decompostion always exists, even if the matrix does not have
 full rank, so the constructor will never fail.  The primary use of the
 QR decomposition is in the least squares solution of nonsquare systems
 of simultaneous linear equations.  This will fail if <tt>isFullRank()</tt>
 returns <tt>false</tt>.
 */

/** partially deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
public class QRDecomposition {

  /** Array for internal storage of decomposition. */
  private final Matrix QR;

  /** Row and column dimensions. */
  private final int originalRows, originalColumns;

  /** Array for internal storage of diagonal of R. */
  private final Vector Rdiag;

  /**
   * Constructs and returns a new QR decomposition object;  computed by Householder reflections; The decomposed matrices
   * can be retrieved via instance methods of the returned decomposition object.
   *
   * @param A A rectangular matrix.
   * @throws IllegalArgumentException if <tt>A.rows() < A.columns()</tt>.
   */

  public QRDecomposition(Matrix A) {

    // Initialize.
    QR = A.clone();
    originalRows = A.numRows();
    originalColumns = A.numCols();
    Rdiag = new DenseVector(originalColumns);

    // precompute and cache some views to avoid regenerating them time and again
    Vector[] QRcolumnsPart = new Vector[originalColumns];
    for (int k = 0; k < originalColumns; k++) {
      QRcolumnsPart[k] = QR.viewColumn(k).viewPart(k, originalRows - k);
    }

    // Main loop.
    for (int k = 0; k < originalColumns; k++) {
      //DoubleMatrix1D QRcolk = QR.viewColumn(k).viewPart(k,m-k);
      // Compute 2-norm of k-th column without under/overflow.
      double nrm = 0;
      //if (k<m) nrm = QRcolumnsPart[k].aggregate(hypot,F.identity);

      for (int i = k; i < originalRows; i++) { // fixes bug reported by hong.44@osu.edu
        nrm = Algebra.hypot(nrm, QR.getQuick(i, k));
      }


      if (nrm != 0.0) {
        // Form k-th Householder vector.
        if (QR.getQuick(k, k) < 0) {
          nrm = -nrm;
        }
        QRcolumnsPart[k].assign(Functions.div(nrm));
        /*
        for (int i = k; i < m; i++) {
           QR[i][k] /= nrm;
        }
        */

        QR.setQuick(k, k, QR.getQuick(k, k) + 1);

        // Apply transformation to remaining columns.
        for (int j = k + 1; j < originalColumns; j++) {
          Vector QRcolj = QR.viewColumn(j).viewPart(k, originalRows - k);
          double s = QRcolumnsPart[k].dot(QRcolj);
          /*
          // fixes bug reported by John Chambers
          DoubleMatrix1D QRcolj = QR.viewColumn(j).viewPart(k,m-k);
          double s = QRcolumnsPart[k].zDotProduct(QRcolumns[j]);
          double s = 0.0;
          for (int i = k; i < m; i++) {
            s += QR[i][k]*QR[i][j];
          }
          */
          s = -s / QR.getQuick(k, k);
          //QRcolumnsPart[j].assign(QRcolumns[k], F.plusMult(s));

          for (int i = k; i < originalRows; i++) {
            QR.setQuick(i, j, QR.getQuick(i, j) + s * QR.getQuick(i, k));
          }

        }
      }
      Rdiag.setQuick(k, -nrm);
    }
  }

  /**
   * Returns the Householder vectors <tt>H</tt>.  Not covered by tests yet.
   *
   * @return A lower trapezoidal matrix whose columns define the householder reflections.
   *
   */
  @Deprecated
  public Matrix getH() {
    Matrix H = QR.clone();
    int rows = H.numRows();
    int columns = H.numCols();
    for (int i = 0; i < rows; i++) {
      for (int j = i + 1; j < columns; j++) {
        H.setQuick(i, j, 0);
      }
    }
    return H;
  }

  /**
   * Generates and returns the (economy-sized) orthogonal factor <tt>Q</tt>.
   *
   * @return <tt>Q</tt>
   */
  public Matrix getQ() {
    int columns = Math.min(originalColumns, originalRows);
    Matrix Q = QR.like(originalRows, columns);
    for (int k = columns - 1; k >= 0; k--) {
      Vector QRcolk = QR.viewColumn(k).viewPart(k, originalRows - k);
      Q.set(k, k, 1);
      for (int j = k; j < columns; j++) {
        if (QR.get(k, k) != 0) {
          Vector Qcolj = Q.viewColumn(j).viewPart(k, originalRows - k);
          double s = -QRcolk.dot(Qcolj) / QR.get(k, k);
          Qcolj.assign(QRcolk, Functions.plusMult(s));
        }
      }
    }
    return Q;
  }

  /**
   * Returns the upper triangular factor, <tt>R</tt>.
   *
   * @return <tt>R</tt>
   */
  public Matrix getR() {
    int rows = Math.min(originalRows, originalColumns);
    Matrix R = QR.like(rows, originalColumns);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < originalColumns; j++) {
        if (i < j) {
          R.setQuick(i, j, QR.getQuick(i, j));
        } else if (i == j) {
          R.setQuick(i, j, Rdiag.getQuick(i));
        } else {
          R.setQuick(i, j, 0);
        }
      }
    }
    return R;
  }

  /**
   * Returns whether the matrix <tt>A</tt> has full rank.
   *
   * @return true if <tt>R</tt>, and hence <tt>A</tt>, has full rank.
   */
  public boolean hasFullRank() {
    for (int j = 0; j < originalColumns; j++) {
      if (Rdiag.getQuick(j) == 0) {
        return false;
      }
    }
    return true;
  }

  /**
   * Least squares solution of <tt>A*X = B</tt>; <tt>returns X</tt>.
   *
   * @param B A matrix with as many rows as <tt>A</tt> and any number of columns.
   * @return <tt>X</tt> that minimizes the two norm of <tt>Q*R*X - B</tt>.
   * @throws IllegalArgumentException if <tt>B.rows() != A.rows()</tt>.
   */
  public Matrix solve(Matrix B) {
    if (B.numRows() != originalRows) {
      throw new IllegalArgumentException("Matrix row dimensions must agree.");
    }

    int columns = B.numCols();
    Matrix X = B.like(originalColumns, columns);

    // this can all be done a bit more efficiently if we don't actually
    // form explicit versions of Q^T and R but this code isn't soo bad
    // and it is much easier to understand
    Matrix Qt = getQ().transpose();
    Matrix Y = Qt.times(B);

    Matrix R = getR();
    for (int k = Math.min(originalColumns, originalRows) - 1; k >= 0; k--) {
      // X[k,] = Y[k,] / R[k,k], note that X[k,] starts with 0 so += is same as =
      X.viewRow(k).assign(Y.viewRow(k), Functions.plusMult(1 / R.get(k, k)));

      // Y[0:(k-1),] -= R[0:(k-1),k] * X[k,]
      Vector rColumn = R.viewColumn(k).viewPart(0, k);
      for (int c = 0; c < columns; c++) {
        Y.viewColumn(c).viewPart(0, k).assign(rColumn, Functions.plusMult(-X.get(k, c)));
      }
    }
    return X;
  }

  /**
   * Returns a rough string rendition of a QR.
   */
  public String toString() {
    return String.format("QR(%d,%d,fullRank=%s)", originalColumns, originalRows, hasFullRank());
  }
}
