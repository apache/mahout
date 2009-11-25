/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.linalg;

import org.apache.mahout.matrix.matrix.DoubleMatrix1D;
import org.apache.mahout.matrix.matrix.DoubleMatrix2D;
/**
 A low level version of {@link LUDecomposition}, avoiding unnecessary memory allocation and copying.
 The input to <tt>decompose</tt> methods is overriden with the result (LU).
 The input to <tt>solve</tt> methods is overriden with the result (X).
 In addition to <tt>LUDecomposition</tt>, this class also includes a faster variant of the decomposition, specialized for tridiagonal (and hence also diagonal) matrices,
 as well as a solver tuned for vectors.
 Its disadvantage is that it is a bit more difficult to use than <tt>LUDecomposition</tt>.
 Thus, you may want to disregard this class and come back later, if a need for speed arises.
 <p>
 An instance of this class remembers the result of its last decomposition.
 Usage pattern is as follows: Create an instance of this class, call a decompose method,
 then retrieve the decompositions, determinant, and/or solve as many equation problems as needed.
 Once another matrix needs to be LU-decomposed, you need not create a new instance of this class.
 Start again by calling a decompose method, then retrieve the decomposition and/or solve your equations, and so on.
 In case a <tt>LU</tt> matrix is already available, call method <tt>setLU</tt> instead of <tt>decompose</tt> and proceed with solving et al.
 <p>
 If a matrix shall not be overriden, use <tt>matrix.copy()</tt> and hand the the copy to methods.
 <p>
 For an <tt>m x n</tt> matrix <tt>A</tt> with <tt>m >= n</tt>, the LU decomposition is an <tt>m x n</tt>
 unit lower triangular matrix <tt>L</tt>, an <tt>n x n</tt> upper triangular matrix <tt>U</tt>,
 and a permutation vector <tt>piv</tt> of length <tt>m</tt> so that <tt>A(piv,:) = L*U</tt>;
 If <tt>m < n</tt>, then <tt>L</tt> is <tt>m x m</tt> and <tt>U</tt> is <tt>m x n</tt>.
 <P>
 The LU decomposition with pivoting always exists, even if the matrix is
 singular, so the decompose methods will never fail.  The primary use of the
 LU decomposition is in the solution of square systems of simultaneous
 linear equations.
 Attempting to solve such a system will throw an exception if <tt>isNonsingular()</tt> returns false.
 <p>
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class LUDecompositionQuick implements java.io.Serializable {

  /** Array for internal storage of decomposition. */
  protected DoubleMatrix2D LU;

  /** pivot sign. */
  protected int pivsign;

  /** Internal storage of pivot vector. */
  protected int[] piv;

  protected boolean isNonSingular;

  protected Algebra algebra;

  protected transient double[] workDouble;
  protected transient int[] work1;
  protected transient int[] work2;

  /**
   * Constructs and returns a new LU Decomposition object with default tolerance <tt>1.0E-9</tt> for singularity
   * detection.
   */
  public LUDecompositionQuick() {
    this(Property.DEFAULT.tolerance());
  }

  /** Constructs and returns a new LU Decomposition object which uses the given tolerance for singularity detection; */
  public LUDecompositionQuick(double tolerance) {
    this.algebra = new Algebra(tolerance);
  }

  /**
   * Decomposes matrix <tt>A</tt> into <tt>L</tt> and <tt>U</tt> (in-place). Upon return <tt>A</tt> is overridden with
   * the result <tt>LU</tt>, such that <tt>L*U = A</tt>. Uses a "left-looking", dot-product, Crout/Doolittle algorithm.
   *
   * @param A any matrix.
   */
  public void decompose(DoubleMatrix2D A) {
    // setup
    LU = A;
    int m = A.rows();
    int n = A.columns();

    // setup pivot vector
    if (this.piv == null || this.piv.length != m) {
      this.piv = new int[m];
    }
    for (int i = m; --i >= 0;) {
      piv[i] = i;
    }
    pivsign = 1;

    if (m * n == 0) {
      setLU(LU);
      return; // nothing to do
    }

    //precompute and cache some views to avoid regenerating them time and again
    DoubleMatrix1D[] LUrows = new DoubleMatrix1D[m];
    for (int i = 0; i < m; i++) {
      LUrows[i] = LU.viewRow(i);
    }

    org.apache.mahout.matrix.list.IntArrayList nonZeroIndexes =
        new org.apache.mahout.matrix.list.IntArrayList(); // sparsity
    DoubleMatrix1D LUcolj = LU.viewColumn(0).like();  // blocked column j
    org.apache.mahout.jet.math.Mult multFunction = org.apache.mahout.jet.math.Mult.mult(0);

    // Outer loop.
    int CUT_OFF = 10;
    for (int j = 0; j < n; j++) {
      // blocking (make copy of j-th column to localize references)
      LUcolj.assign(LU.viewColumn(j));

      // sparsity detection
      int maxCardinality = m / CUT_OFF; // == heuristic depending on speedup
      LUcolj.getNonZeros(nonZeroIndexes, null, maxCardinality);
      int cardinality = nonZeroIndexes.size();
      boolean sparse = (cardinality < maxCardinality);

      // Apply previous transformations.
      for (int i = 0; i < m; i++) {
        int kmax = Math.min(i, j);
        double s;
        if (sparse) {
          s = LUrows[i].zDotProduct(LUcolj, 0, kmax, nonZeroIndexes);
        } else {
          s = LUrows[i].zDotProduct(LUcolj, 0, kmax);
        }
        double before = LUcolj.getQuick(i);
        double after = before - s;
        LUcolj.setQuick(i, after); // LUcolj is a copy
        LU.setQuick(i, j, after);   // this is the original
        if (sparse) {
          if (before == 0 && after != 0) { // nasty bug fixed!
            int pos = nonZeroIndexes.binarySearch(i);
            pos = -pos - 1;
            nonZeroIndexes.beforeInsert(pos, i);
          }
          if (before != 0 && after == 0) {
            nonZeroIndexes.remove(nonZeroIndexes.binarySearch(i));
          }
        }
      }

      // Find pivot and exchange if necessary.
      int p = j;
      if (p < m) {
        double max = Math.abs(LUcolj.getQuick(p));
        for (int i = j + 1; i < m; i++) {
          double v = Math.abs(LUcolj.getQuick(i));
          if (v > max) {
            p = i;
            max = v;
          }
        }
      }
      if (p != j) {
        LUrows[p].swap(LUrows[j]);
        int k = piv[p];
        piv[p] = piv[j];
        piv[j] = k;
        pivsign = -pivsign;
      }

      // Compute multipliers.
      double jj;
      if (j < m && (jj = LU.getQuick(j, j)) != 0.0) {
        multFunction.setMultiplicator(1 / jj);
        LU.viewColumn(j).viewPart(j + 1, m - (j + 1)).assign(multFunction);
      }

    }
    setLU(LU);
  }

  /**
   * Decomposes the banded and square matrix <tt>A</tt> into <tt>L</tt> and <tt>U</tt> (in-place). Upon return
   * <tt>A</tt> is overridden with the result <tt>LU</tt>, such that <tt>L*U = A</tt>. Currently supports diagonal and
   * tridiagonal matrices, all other cases fall through to {@link #decompose(DoubleMatrix2D)}.
   *
   * @param semiBandwidth == 1 --> A is diagonal, == 2 --> A is tridiagonal.
   * @param A             any matrix.
   */
  public void decompose(DoubleMatrix2D A, int semiBandwidth) {
    if (!algebra.property().isSquare(A) || semiBandwidth < 0 || semiBandwidth > 2) {
      decompose(A);
      return;
    }
    // setup
    LU = A;
    int m = A.rows();
    int n = A.columns();

    // setup pivot vector
    if (this.piv == null || this.piv.length != m) {
      this.piv = new int[m];
    }
    for (int i = m; --i >= 0;) {
      piv[i] = i;
    }
    pivsign = 1;

    if (m * n == 0) {
      setLU(A);
      return; // nothing to do
    }

    //if (semiBandwidth == 1) { // A is diagonal; nothing to do
    if (semiBandwidth == 2) { // A is tridiagonal
      // currently no pivoting !
      if (n > 1) {
        A.setQuick(1, 0, A.getQuick(1, 0) / A.getQuick(0, 0));
      }

      for (int i = 1; i < n; i++) {
        double ei = A.getQuick(i, i) - A.getQuick(i, i - 1) * A.getQuick(i - 1, i);
        A.setQuick(i, i, ei);
        if (i < n - 1) {
          A.setQuick(i + 1, i, A.getQuick(i + 1, i) / ei);
        }
      }
    }
    setLU(A);
  }

  /**
   * Returns the determinant, <tt>det(A)</tt>.
   *
   * @throws IllegalArgumentException if <tt>A.rows() != A.columns()</tt> (Matrix must be square).
   */
  public double det() {
    int m = m();
    int n = n();
    if (m != n) {
      throw new IllegalArgumentException("Matrix must be square.");
    }

    if (!isNonsingular()) {
      return 0;
    } // avoid rounding errors

    double det = (double) pivsign;
    for (int j = 0; j < n; j++) {
      det *= LU.getQuick(j, j);
    }
    return det;
  }

  /**
   * Returns pivot permutation vector as a one-dimensional double array
   *
   * @return (double) piv
   */
  protected double[] getDoublePivot() {
    int m = m();
    double[] vals = new double[m];
    for (int i = 0; i < m; i++) {
      vals[i] = (double) piv[i];
    }
    return vals;
  }

  /**
   * Returns the lower triangular factor, <tt>L</tt>.
   *
   * @return <tt>L</tt>
   */
  public DoubleMatrix2D getL() {
    return lowerTriangular(LU.copy());
  }

  /**
   * Returns a copy of the combined lower and upper triangular factor, <tt>LU</tt>.
   *
   * @return <tt>LU</tt>
   */
  public DoubleMatrix2D getLU() {
    return LU.copy();
  }

  /**
   * Returns the pivot permutation vector (not a copy of it).
   *
   * @return piv
   */
  public int[] getPivot() {
    return piv;
  }

  /**
   * Returns the upper triangular factor, <tt>U</tt>.
   *
   * @return <tt>U</tt>
   */
  public DoubleMatrix2D getU() {
    return upperTriangular(LU.copy());
  }

  /**
   * Returns whether the matrix is nonsingular (has an inverse).
   *
   * @return true if <tt>U</tt>, and hence <tt>A</tt>, is nonsingular; false otherwise.
   */
  public boolean isNonsingular() {
    return isNonSingular;
  }

  /**
   * Returns whether the matrix is nonsingular.
   *
   * @return true if <tt>matrix</tt> is nonsingular; false otherwise.
   */
  protected boolean isNonsingular(DoubleMatrix2D matrix) {
    int m = matrix.rows();
    int n = matrix.columns();
    double epsilon = algebra.property().tolerance(); // consider numerical instability
    for (int j = Math.min(n, m); --j >= 0;) {
      //if (matrix.getQuick(j,j) == 0) return false;
      if (Math.abs(matrix.getQuick(j, j)) <= epsilon) {
        return false;
      }
    }
    return true;
  }

  /**
   * Modifies the matrix to be a lower triangular matrix. <p> <b>Examples:</b> <table border="0"> <tr nowrap> <td
   * valign="top">3 x 5 matrix:<br> 9, 9, 9, 9, 9<br> 9, 9, 9, 9, 9<br> 9, 9, 9, 9, 9 </td> <td
   * align="center">triang.Upper<br> ==></td> <td valign="top">3 x 5 matrix:<br> 9, 9, 9, 9, 9<br> 0, 9, 9, 9, 9<br> 0,
   * 0, 9, 9, 9</td> </tr> <tr nowrap> <td valign="top">5 x 3 matrix:<br> 9, 9, 9<br> 9, 9, 9<br> 9, 9, 9<br> 9, 9,
   * 9<br> 9, 9, 9 </td> <td align="center">triang.Upper<br> ==></td> <td valign="top">5 x 3 matrix:<br> 9, 9, 9<br> 0,
   * 9, 9<br> 0, 0, 9<br> 0, 0, 0<br> 0, 0, 0</td> </tr> <tr nowrap> <td valign="top">3 x 5 matrix:<br> 9, 9, 9, 9,
   * 9<br> 9, 9, 9, 9, 9<br> 9, 9, 9, 9, 9 </td> <td align="center">triang.Lower<br> ==></td> <td valign="top">3 x 5
   * matrix:<br> 1, 0, 0, 0, 0<br> 9, 1, 0, 0, 0<br> 9, 9, 1, 0, 0</td> </tr> <tr nowrap> <td valign="top">5 x 3
   * matrix:<br> 9, 9, 9<br> 9, 9, 9<br> 9, 9, 9<br> 9, 9, 9<br> 9, 9, 9 </td> <td align="center">triang.Lower<br>
   * ==></td> <td valign="top">5 x 3 matrix:<br> 1, 0, 0<br> 9, 1, 0<br> 9, 9, 1<br> 9, 9, 9<br> 9, 9, 9</td> </tr>
   * </table>
   *
   * @return <tt>A</tt> (for convenience only).
   */
  protected DoubleMatrix2D lowerTriangular(DoubleMatrix2D A) {
    int rows = A.rows();
    int columns = A.columns();
    int min = Math.min(rows, columns);
    for (int r = min; --r >= 0;) {
      for (int c = min; --c >= 0;) {
        if (r < c) {
          A.setQuick(r, c, 0);
        } else if (r == c) {
          A.setQuick(r, c, 1);
        }
      }
    }
    if (columns > rows) {
      A.viewPart(0, min, rows, columns - min).assign(0);
    }

    return A;
  }

  /**
   *
   */
  protected int m() {
    return LU.rows();
  }

  /**
   *
   */
  protected int n() {
    return LU.columns();
  }

  /**
   * Sets the combined lower and upper triangular factor, <tt>LU</tt>. The parameter is not checked; make sure it is
   * indeed a proper LU decomposition.
   */
  public void setLU(DoubleMatrix2D LU) {
    this.LU = LU;
    this.isNonSingular = isNonsingular(LU);
  }

  /**
   * Solves the system of equations <tt>A*X = B</tt> (in-place). Upon return <tt>B</tt> is overridden with the result
   * <tt>X</tt>, such that <tt>L*U*X = B(piv)</tt>.
   *
   * @param B A vector with <tt>B.size() == A.rows()</tt>.
   * @throws IllegalArgumentException if </tt>B.size() != A.rows()</tt>.
   * @throws IllegalArgumentException if A is singular, that is, if <tt>!isNonsingular()</tt>.
   * @throws IllegalArgumentException if <tt>A.rows() < A.columns()</tt>.
   */
  public void solve(DoubleMatrix1D B) {
    algebra.property().checkRectangular(LU);
    int m = m();
    int n = n();
    if (B.size() != m) {
      throw new IllegalArgumentException("Matrix dimensions must agree.");
    }
    if (!this.isNonsingular()) {
      throw new IllegalArgumentException("Matrix is singular.");
    }


    // right hand side with pivoting
    // Matrix Xmat = B.getMatrix(piv,0,nx-1);
    if (this.workDouble == null || this.workDouble.length < m) {
      this.workDouble = new double[m];
    }
    algebra.permute(B, this.piv, this.workDouble);

    if (m * n == 0) {
      return;
    } // nothing to do

    // Solve L*Y = B(piv,:)
    for (int k = 0; k < n; k++) {
      double f = B.getQuick(k);
      if (f != 0) {
        for (int i = k + 1; i < n; i++) {
          // B[i] -= B[k]*LU[i][k];
          double v = LU.getQuick(i, k);
          if (v != 0) {
            B.setQuick(i, B.getQuick(i) - f * v);
          }
        }
      }
    }

    // Solve U*B = Y;
    for (int k = n - 1; k >= 0; k--) {
      // B[k] /= LU[k,k]
      B.setQuick(k, B.getQuick(k) / LU.getQuick(k, k));
      double f = B.getQuick(k);
      if (f != 0) {
        for (int i = 0; i < k; i++) {
          // B[i] -= B[k]*LU[i][k];
          double v = LU.getQuick(i, k);
          if (v != 0) {
            B.setQuick(i, B.getQuick(i) - f * v);
          }
        }
      }
    }
  }

  /**
   * Solves the system of equations <tt>A*X = B</tt> (in-place). Upon return <tt>B</tt> is overridden with the result
   * <tt>X</tt>, such that <tt>L*U*X = B(piv,:)</tt>.
   *
   * @param B A matrix with as many rows as <tt>A</tt> and any number of columns.
   * @throws IllegalArgumentException if </tt>B.rows() != A.rows()</tt>.
   * @throws IllegalArgumentException if A is singular, that is, if <tt>!isNonsingular()</tt>.
   * @throws IllegalArgumentException if <tt>A.rows() < A.columns()</tt>.
   */
  public void solve(DoubleMatrix2D B) {
    algebra.property().checkRectangular(LU);
    int m = m();
    int n = n();
    if (B.rows() != m) {
      throw new IllegalArgumentException("Matrix row dimensions must agree.");
    }
    if (!this.isNonsingular()) {
      throw new IllegalArgumentException("Matrix is singular.");
    }


    // right hand side with pivoting
    // Matrix Xmat = B.getMatrix(piv,0,nx-1);
    if (this.work1 == null || this.work1.length < m) {
      this.work1 = new int[m];
    }
    //if (this.work2 == null || this.work2.length < m) this.work2 = new int[m];
    algebra.permuteRows(B, this.piv, this.work1);

    if (m * n == 0) {
      return;
    } // nothing to do
    int nx = B.columns();

    //precompute and cache some views to avoid regenerating them time and again
    DoubleMatrix1D[] Brows = new DoubleMatrix1D[n];
    for (int k = 0; k < n; k++) {
      Brows[k] = B.viewRow(k);
    }

    // transformations
    org.apache.mahout.jet.math.Mult div = org.apache.mahout.jet.math.Mult.div(0);
    org.apache.mahout.jet.math.PlusMult minusMult = org.apache.mahout.jet.math.PlusMult.minusMult(0);

    org.apache.mahout.matrix.list.IntArrayList nonZeroIndexes =
        new org.apache.mahout.matrix.list.IntArrayList(); // sparsity
    DoubleMatrix1D Browk = org.apache.mahout.matrix.matrix.DoubleFactory1D.dense.make(nx); // blocked row k

    // Solve L*Y = B(piv,:)
    int CUT_OFF = 10;
    for (int k = 0; k < n; k++) {
      // blocking (make copy of k-th row to localize references)
      Browk.assign(Brows[k]);

      // sparsity detection
      int maxCardinality = nx / CUT_OFF; // == heuristic depending on speedup
      Browk.getNonZeros(nonZeroIndexes, null, maxCardinality);
      int cardinality = nonZeroIndexes.size();
      boolean sparse = (cardinality < maxCardinality);

      for (int i = k + 1; i < n; i++) {
        //for (int j = 0; j < nx; j++) B[i][j] -= B[k][j]*LU[i][k];
        //for (int j = 0; j < nx; j++) B.set(i,j, B.get(i,j) - B.get(k,j)*LU.get(i,k));

        minusMult.setMultiplicator(-LU.getQuick(i, k));
        if (minusMult.getMultiplicator() != 0) {
          if (sparse) {
            Brows[i].assign(Browk, minusMult, nonZeroIndexes);
          } else {
            Brows[i].assign(Browk, minusMult);
          }
        }
      }
    }

    // Solve U*B = Y;
    for (int k = n - 1; k >= 0; k--) {
      // for (int j = 0; j < nx; j++) B[k][j] /= LU[k][k];
      // for (int j = 0; j < nx; j++) B.set(k,j, B.get(k,j) / LU.get(k,k));
      div.setMultiplicator(1 / LU.getQuick(k, k));
      Brows[k].assign(div);

      // blocking
      if (Browk == null) {
        Browk = org.apache.mahout.matrix.matrix.DoubleFactory1D.dense.make(B.columns());
      }
      Browk.assign(Brows[k]);

      // sparsity detection
      int maxCardinality = nx / CUT_OFF; // == heuristic depending on speedup
      Browk.getNonZeros(nonZeroIndexes, null, maxCardinality);
      int cardinality = nonZeroIndexes.size();
      boolean sparse = (cardinality < maxCardinality);

      //Browk.getNonZeros(nonZeroIndexes,null);
      //boolean sparse = nonZeroIndexes.size() < nx/10;

      for (int i = 0; i < k; i++) {
        // for (int j = 0; j < nx; j++) B[i][j] -= B[k][j]*LU[i][k];
        // for (int j = 0; j < nx; j++) B.set(i,j, B.get(i,j) - B.get(k,j)*LU.get(i,k));

        minusMult.setMultiplicator(-LU.getQuick(i, k));
        if (minusMult.getMultiplicator() != 0) {
          if (sparse) {
            Brows[i].assign(Browk, minusMult, nonZeroIndexes);
          } else {
            Brows[i].assign(Browk, minusMult);
          }
        }
      }
    }
  }

  /**
   * Solves <tt>A*X = B</tt>.
   *
   * @param B A matrix with as many rows as <tt>A</tt> and any number of columns.
   * @return <tt>X</tt> so that <tt>L*U*X = B(piv,:)</tt>.
   * @throws IllegalArgumentException if </tt>B.rows() != A.rows()</tt>.
   * @throws IllegalArgumentException if A is singular, that is, if <tt>!this.isNonsingular()</tt>.
   * @throws IllegalArgumentException if <tt>A.rows() < A.columns()</tt>.
   */
  private void solveOld(DoubleMatrix2D B) {
    algebra.property().checkRectangular(LU);
    int m = m();
    int n = n();
    if (B.rows() != m) {
      throw new IllegalArgumentException("Matrix row dimensions must agree.");
    }
    if (!this.isNonsingular()) {
      throw new IllegalArgumentException("Matrix is singular.");
    }

    // Copy right hand side with pivoting
    int nx = B.columns();

    if (this.work1 == null || this.work1.length < m) {
      this.work1 = new int[m];
    }
    //if (this.work2 == null || this.work2.length < m) this.work2 = new int[m];
    algebra.permuteRows(B, this.piv, this.work1);

    // Solve L*Y = B(piv,:) --> Y (Y is modified B)
    for (int k = 0; k < n; k++) {
      for (int i = k + 1; i < n; i++) {
        double mult = LU.getQuick(i, k);
        if (mult != 0) {
          for (int j = 0; j < nx; j++) {
            //B[i][j] -= B[k][j]*LU[i,k];
            B.setQuick(i, j, B.getQuick(i, j) - B.getQuick(k, j) * mult);
          }
        }
      }
    }
    // Solve U*X = Y; --> X (X is modified B)
    for (int k = n - 1; k >= 0; k--) {
      double mult = 1 / LU.getQuick(k, k);
      if (mult != 1) {
        for (int j = 0; j < nx; j++) {
          //B[k][j] /= LU[k][k];
          B.setQuick(k, j, B.getQuick(k, j) * mult);
        }
      }
      for (int i = 0; i < k; i++) {
        mult = LU.getQuick(i, k);
        if (mult != 0) {
          for (int j = 0; j < nx; j++) {
            //B[i][j] -= B[k][j]*LU[i][k];
            B.setQuick(i, j, B.getQuick(i, j) - B.getQuick(k, j) * mult);
          }
        }
      }
    }
  }

  /**
   * Returns a String with (propertyName, propertyValue) pairs. Useful for debugging or to quickly get the rough
   * picture. For example,
   * <pre>
   * rank          : 3
   * trace         : 0
   * </pre>
   */
  public String toString() {
    StringBuilder buf = new StringBuilder();

    buf.append("-----------------------------------------------------------------------------\n");
    buf.append("LUDecompositionQuick(A) --> isNonSingular(A), det(A), pivot, L, U, inverse(A)\n");
    buf.append("-----------------------------------------------------------------------------\n");

    buf.append("isNonSingular = ");
    String unknown = "Illegal operation or error: ";
    try {
      buf.append(String.valueOf(this.isNonsingular()));
    }
    catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }

    buf.append("\ndet = ");
    try {
      buf.append(String.valueOf(this.det()));
    }
    catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }

    buf.append("\npivot = ");
    try {
      buf.append(String.valueOf(new org.apache.mahout.matrix.list.IntArrayList(this.getPivot())));
    }
    catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }

    buf.append("\n\nL = ");
    try {
      buf.append(String.valueOf(this.getL()));
    }
    catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }

    buf.append("\n\nU = ");
    try {
      buf.append(String.valueOf(this.getU()));
    }
    catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }

    buf.append("\n\ninverse(A) = ");
    DoubleMatrix2D identity = org.apache.mahout.matrix.matrix.DoubleFactory2D.dense.identity(LU.rows());
    try {
      this.solve(identity);
      buf.append(String.valueOf(identity));
    }
    catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }

    return buf.toString();
  }

  /**
   * Modifies the matrix to be an upper triangular matrix.
   *
   * @return <tt>A</tt> (for convenience only).
   */
  protected DoubleMatrix2D upperTriangular(DoubleMatrix2D A) {
    int rows = A.rows();
    int columns = A.columns();
    int min = Math.min(rows, columns);
    for (int r = min; --r >= 0;) {
      for (int c = min; --c >= 0;) {
        if (r > c) {
          A.setQuick(r, c, 0);
        }
      }
    }
    if (columns < rows) {
      A.viewPart(min, 0, rows - min, columns).assign(0);
    }

    return A;
  }

  /**
   * Returns pivot permutation vector as a one-dimensional double array
   *
   * @return (double) piv
   */
  private double[] xgetDoublePivot() {
    int m = m();
    double[] vals = new double[m];
    for (int i = 0; i < m; i++) {
      vals[i] = (double) piv[i];
    }
    return vals;
  }
}
