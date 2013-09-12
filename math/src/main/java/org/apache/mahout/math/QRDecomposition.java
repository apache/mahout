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
 The QR decomposition always exists, even if the matrix does not have
 full rank, so the constructor will never fail.  The primary use of the
 QR decomposition is in the least squares solution of non-square systems
 of simultaneous linear equations.  This will fail if <tt>isFullRank()</tt>
 returns <tt>false</tt>.
 */

public class QRDecomposition implements QR {
  private final Matrix q;
  private final Matrix r;
  private final boolean fullRank;
  private final int rows;
  private final int columns;

  /**
   * Constructs and returns a new QR decomposition object;  computed by Householder reflections; The
   * decomposed matrices can be retrieved via instance methods of the returned decomposition
   * object.
   *
   * @param a A rectangular matrix.
   * @throws IllegalArgumentException if <tt>A.rows() < A.columns()</tt>.
   */
  public QRDecomposition(Matrix a) {

    rows = a.rowSize();
    int min = Math.min(a.rowSize(), a.columnSize());
    columns = a.columnSize();

    Matrix qTmp = a.clone();

    boolean fullRank = true;

    r = new DenseMatrix(min, columns);

    for (int i = 0; i < min; i++) {
      Vector qi = qTmp.viewColumn(i);
      double alpha = qi.norm(2);
      if (Math.abs(alpha) > Double.MIN_VALUE) {
        qi.assign(Functions.div(alpha));
      } else {
        if (Double.isInfinite(alpha) || Double.isNaN(alpha)) {
          throw new ArithmeticException("Invalid intermediate result");
        }
        fullRank = false;
      }
      r.set(i, i, alpha);

      for (int j = i + 1; j < columns; j++) {
        Vector qj = qTmp.viewColumn(j);
        double norm = qj.norm(2);
        if (Math.abs(norm) > Double.MIN_VALUE) {
          double beta = qi.dot(qj);
          r.set(i, j, beta);
          if (j < min) {
            qj.assign(qi, Functions.plusMult(-beta));
          }
        } else {
          if (Double.isInfinite(norm) || Double.isNaN(norm)) {
            throw new ArithmeticException("Invalid intermediate result");
          }
        }
      }
    }
    if (columns > min) {
      q = qTmp.viewPart(0, rows, 0, min).clone();
    } else {
      q = qTmp;
    }
    this.fullRank = fullRank;
  }

  /**
   * Generates and returns the (economy-sized) orthogonal factor <tt>Q</tt>.
   *
   * @return <tt>Q</tt>
   */
  @Override
  public Matrix getQ() {
    return q;
  }

  /**
   * Returns the upper triangular factor, <tt>R</tt>.
   *
   * @return <tt>R</tt>
   */
  @Override
  public Matrix getR() {
    return r;
  }

  /**
   * Returns whether the matrix <tt>A</tt> has full rank.
   *
   * @return true if <tt>R</tt>, and hence <tt>A</tt>, has full rank.
   */
  @Override
  public boolean hasFullRank() {
    return fullRank;
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
    if (B.numRows() != rows) {
      throw new IllegalArgumentException("Matrix row dimensions must agree.");
    }

    int cols = B.numCols();
    Matrix x = B.like(columns, cols);

    // this can all be done a bit more efficiently if we don't actually
    // form explicit versions of Q^T and R but this code isn't so bad
    // and it is much easier to understand
    Matrix qt = getQ().transpose();
    Matrix y = qt.times(B);

    Matrix r = getR();
    for (int k = Math.min(columns, rows) - 1; k >= 0; k--) {
      // X[k,] = Y[k,] / R[k,k], note that X[k,] starts with 0 so += is same as =
      x.viewRow(k).assign(y.viewRow(k), Functions.plusMult(1 / r.get(k, k)));

      // Y[0:(k-1),] -= R[0:(k-1),k] * X[k,]
      Vector rColumn = r.viewColumn(k).viewPart(0, k);
      for (int c = 0; c < cols; c++) {
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
    return String.format(Locale.ENGLISH, "QR(%d x %d,fullRank=%s)", rows, columns, hasFullRank());
  }
}
