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
 *
 * Copyright 1999 CERN - European Organization for Nuclear Research.
 * Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
 * is hereby granted without fee, provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear in supporting documentation.
 * CERN makes no representations about the suitability of this software for any purpose.
 * It is provided "as is" without expressed or implied warranty.
 */
package org.apache.mahout.math;

import org.apache.mahout.math.function.Functions;

import java.util.Locale;


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
public class OldQRDecomposition implements QR {

  /** Array for internal storage of decomposition. */
  private final Matrix qr;

  /** Row and column dimensions. */
  private final int originalRows;
  private final int originalColumns;

  /** Array for internal storage of diagonal of R. */
  private final Vector rDiag;

  /**
   * Constructs and returns a new QR decomposition object;  computed by Householder reflections; The decomposed matrices
   * can be retrieved via instance methods of the returned decomposition object.
   *
   * @param a A rectangular matrix.
   * @throws IllegalArgumentException if <tt>A.rows() < A.columns()</tt>.
   */

  public OldQRDecomposition(Matrix a) {

    // Initialize.
    qr = a.clone();
    originalRows = a.numRows();
    originalColumns = a.numCols();
    rDiag = new DenseVector(originalColumns);

    // precompute and cache some views to avoid regenerating them time and again
    Vector[] QRcolumnsPart = new Vector[originalColumns];
    for (int k = 0; k < originalColumns; k++) {
      QRcolumnsPart[k] = qr.viewColumn(k).viewPart(k, originalRows - k);
    }

    // Main loop.
    for (int k = 0; k < originalColumns; k++) {
      //DoubleMatrix1D QRcolk = QR.viewColumn(k).viewPart(k,m-k);
      // Compute 2-norm of k-th column without under/overflow.
      double nrm = 0;
      //if (k<m) nrm = QRcolumnsPart[k].aggregate(hypot,F.identity);

      for (int i = k; i < originalRows; i++) { // fixes bug reported by hong.44@osu.edu
        nrm = Algebra.hypot(nrm, qr.getQuick(i, k));
      }


      if (nrm != 0.0) {
        // Form k-th Householder vector.
        if (qr.getQuick(k, k) < 0) {
          nrm = -nrm;
        }
        QRcolumnsPart[k].assign(Functions.div(nrm));
        /*
        for (int i = k; i < m; i++) {
           QR[i][k] /= nrm;
        }
        */

        qr.setQuick(k, k, qr.getQuick(k, k) + 1);

        // Apply transformation to remaining columns.
        for (int j = k + 1; j < originalColumns; j++) {
          Vector QRcolj = qr.viewColumn(j).viewPart(k, originalRows - k);
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
          s = -s / qr.getQuick(k, k);
          //QRcolumnsPart[j].assign(QRcolumns[k], F.plusMult(s));

          for (int i = k; i < originalRows; i++) {
            qr.setQuick(i, j, qr.getQuick(i, j) + s * qr.getQuick(i, k));
          }

        }
      }
      rDiag.setQuick(k, -nrm);
    }
  }

  /**
   * Generates and returns the (economy-sized) orthogonal factor <tt>Q</tt>.
   *
   * @return <tt>Q</tt>
   */
  @Override
  public Matrix getQ() {
    int columns = Math.min(originalColumns, originalRows);
    Matrix q = qr.like(originalRows, columns);
    for (int k = columns - 1; k >= 0; k--) {
      Vector QRcolk = qr.viewColumn(k).viewPart(k, originalRows - k);
      q.set(k, k, 1);
      for (int j = k; j < columns; j++) {
        if (qr.get(k, k) != 0) {
          Vector Qcolj = q.viewColumn(j).viewPart(k, originalRows - k);
          double s = -QRcolk.dot(Qcolj) / qr.get(k, k);
          Qcolj.assign(QRcolk, Functions.plusMult(s));
        }
      }
    }
    return q;
  }

  /**
   * Returns the upper triangular factor, <tt>R</tt>.
   *
   * @return <tt>R</tt>
   */
  @Override
  public Matrix getR() {
    int rows = Math.min(originalRows, originalColumns);
    Matrix r = qr.like(rows, originalColumns);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < originalColumns; j++) {
        if (i < j) {
          r.setQuick(i, j, qr.getQuick(i, j));
        } else if (i == j) {
          r.setQuick(i, j, rDiag.getQuick(i));
        } else {
          r.setQuick(i, j, 0);
        }
      }
    }
    return r;
  }

  /**
   * Returns whether the matrix <tt>A</tt> has full rank.
   *
   * @return true if <tt>R</tt>, and hence <tt>A</tt>, has full rank.
   */
  @Override
  public boolean hasFullRank() {
    for (int j = 0; j < originalColumns; j++) {
      if (rDiag.getQuick(j) == 0) {
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
  @Override
  public Matrix solve(Matrix B) {
    if (B.numRows() != originalRows) {
      throw new IllegalArgumentException("Matrix row dimensions must agree.");
    }

    int columns = B.numCols();
    Matrix x = B.like(originalColumns, columns);

    // this can all be done a bit more efficiently if we don't actually
    // form explicit versions of Q^T and R but this code isn't soo bad
    // and it is much easier to understand
    Matrix qt = getQ().transpose();
    Matrix y = qt.times(B);

    Matrix r = getR();
    for (int k = Math.min(originalColumns, originalRows) - 1; k >= 0; k--) {
      // X[k,] = Y[k,] / R[k,k], note that X[k,] starts with 0 so += is same as =
      x.viewRow(k).assign(y.viewRow(k), Functions.plusMult(1 / r.get(k, k)));

      // Y[0:(k-1),] -= R[0:(k-1),k] * X[k,]
      Vector rColumn = r.viewColumn(k).viewPart(0, k);
      for (int c = 0; c < columns; c++) {
        y.viewColumn(c).viewPart(0, k).assign(rColumn, Functions.plusMult(-x.get(k, c)));
      }
    }
    return x;
  }

  /**
   * Returns a rough string rendition of a QR.
   */
  @Override
  public String toString() {
    return String.format(Locale.ENGLISH, "QR(%d,%d,fullRank=%s)", originalColumns, originalRows, hasFullRank());
  }
}
