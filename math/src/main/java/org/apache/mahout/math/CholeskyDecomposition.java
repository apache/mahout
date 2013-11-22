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

package org.apache.mahout.math;

import com.google.common.base.Preconditions;
import org.apache.mahout.math.function.Functions;

/**
 * Cholesky decomposition shamelessly ported from JAMA.
 * <p/>
 * A Cholesky decomposition of a semi-positive definite matrix A is a lower triangular matrix L such
 * that L L^* = A.  If A is full rank, L is unique.  If A is real, then it must be symmetric and R
 * will also be real.
 */
public class CholeskyDecomposition {
  private final PivotedMatrix L;
  private boolean isPositiveDefinite;

  public CholeskyDecomposition(Matrix a) {
    this(a, true);
  }

  public CholeskyDecomposition(Matrix a, boolean pivot) {
    int rows = a.rowSize();
    L = new PivotedMatrix(new DenseMatrix(rows, rows));

    // must be square
    Preconditions.checkArgument(rows == a.columnSize(), "Must be a Square Matrix");

    if (pivot) {
      decomposeWithPivoting(a);
    } else {
      decompose(a);
    }
  }

  private void decomposeWithPivoting(Matrix a) {
    int n = a.rowSize();
    L.assign(a);

    // pivoted column-wise submatrix cholesky with simple pivoting
    double uberMax = L.viewDiagonal().aggregate(Functions.MAX, Functions.ABS);
    for (int k = 0; k < n; k++) {
      double max = 0;
      int pivot = k;
      for (int j = k; j < n; j++) {
        if (L.get(j, j) > max) {
          max = L.get(j, j);
          pivot = j;
          if (uberMax < Math.abs(max)) {
            uberMax = Math.abs(max);
          }
        }
      }
      L.swap(k, pivot);

      double akk = L.get(k, k);
      double epsilon = 1.0e-10 * Math.max(uberMax, L.viewColumn(k).aggregate(Functions.MAX, Functions.ABS));

      if (akk < -epsilon) {
        // can't have decidedly negative element on diagonal
        throw new IllegalArgumentException("Matrix is not positive semi-definite");
      } else if (akk <= epsilon) {
        // degenerate column case.  Set all to zero
        L.viewColumn(k).assign(0);
        isPositiveDefinite = false;

        // no need to subtract from remaining sub-matrix
      } else {
        // normalize column by diagonal element
        akk = Math.sqrt(Math.max(0, akk));
        L.viewColumn(k).viewPart(k, n - k).assign(Functions.div(akk));
        L.viewColumn(k).viewPart(0, k).assign(0);

        // subtract off scaled version of this column to the right
        for (int j = k + 1; j < n; j++) {
          Vector columnJ = L.viewColumn(j).viewPart(k, n - k);
          Vector columnK = L.viewColumn(k).viewPart(k, n - k);
          columnJ.assign(columnK, Functions.minusMult(columnK.get(j - k)));
        }

      }
    }
  }

  private void decompose(Matrix a) {
    int n = a.rowSize();
    L.assign(a);

    // column-wise submatrix cholesky with simple pivoting
    for (int k = 0; k < n; k++) {

      double akk = L.get(k, k);

      // set upper part of column to 0.
      L.viewColumn(k).viewPart(0, k).assign(0);

      double epsilon = 1.0e-10 * L.viewColumn(k).aggregate(Functions.MAX, Functions.ABS);
      if (akk <= epsilon) {
        // degenerate column case.  Set diagonal to 1, all others to zero
        L.viewColumn(k).viewPart(k, n - k).assign(0);

        isPositiveDefinite = false;

        // no need to subtract from remaining sub-matrix
      } else {
        // normalize column by diagonal element
        akk = Math.sqrt(Math.max(0, akk));
        L.set(k, k, akk);
        L.viewColumn(k).viewPart(k + 1, n - k - 1).assign(Functions.div(akk));

        // now subtract scaled version of column
        for (int j = k + 1; j < n; j++) {
          Vector columnJ = L.viewColumn(j).viewPart(j, n - j);
          Vector columnK = L.viewColumn(k).viewPart(j, n - j);
          columnJ.assign(columnK, Functions.minusMult(L.get(j, k)));
        }
      }
    }
  }

  public boolean isPositiveDefinite() {
    return isPositiveDefinite;
  }

  public Matrix getL() {
    return L.getBase();
  }

  public PivotedMatrix getPermutedL() {
    return L;
  }

  /**
   * @return Returns the permutation of rows and columns that was applied to L
   */
  public int[] getPivot() {
    return L.getRowPivot();
  }

  public int[] getInversePivot() {
    return L.getInverseRowPivot();
  }

  /**
   * Compute inv(L) * z efficiently.
   *
   * @param z
   */
  public Matrix solveLeft(Matrix z) {
    int n = L.columnSize();
    int nx = z.columnSize();

    Matrix X = new DenseMatrix(n, z.columnSize());
    X.assign(z);

    // Solve L*Y = Z using back-substitution
    // note that k and i have to go in a funny order because L is pivoted
    for (int internalK = 0; internalK < n; internalK++) {
      int k = L.rowUnpivot(internalK);
      for (int j = 0; j < nx; j++) {
        for (int internalI = 0; internalI < internalK; internalI++) {
          int i = L.rowUnpivot(internalI);
          X.set(k, j, X.get(k, j) - X.get(i, j) * L.get(k, i));
        }
        if (L.get(k, k) != 0) {
          X.set(k, j, X.get(k, j) / L.get(k, k));
        } else {
          X.set(k, j, 0);
        }
      }
    }
    return X;
  }

  /**
   * Compute z * inv(L') efficiently
   */
  public Matrix solveRight(Matrix z) {
    int n = z.columnSize();
    int nx = z.rowSize();

    Matrix x = new DenseMatrix(z.rowSize(), z.columnSize());
    x.assign(z);

    // Solve Y*L' = Z using back-substitution
    for (int internalK = 0; internalK < n; internalK++) {
      int k = L.rowUnpivot(internalK);
      for (int j = 0; j < nx; j++) {
        for (int internalI = 0; internalI < k; internalI++) {
          int i = L.rowUnpivot(internalI);
          x.set(j, k, x.get(j, k) - x.get(j, i) * L.get(k, i));
          if (Double.isInfinite(x.get(j, k)) || Double.isNaN(x.get(j, k))) {
            throw new IllegalStateException(
                String.format("Invalid value found at %d,%d (should not be possible)", j, k));
          }
        }
        if (L.get(k, k) != 0) {
          x.set(j, k, x.get(j, k) / L.get(k, k));
        } else {
          x.set(j, k, 0);
        }
        if (Double.isInfinite(x.get(j, k)) || Double.isNaN(x.get(j, k))) {
          throw new IllegalStateException(String.format("Invalid value found at %d,%d (should not be possible)", j, k));
        }
      }
    }
    return x;
  }

}

