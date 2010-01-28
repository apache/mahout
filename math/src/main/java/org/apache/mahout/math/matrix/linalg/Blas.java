/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import org.apache.mahout.math.function.BinaryFunction;
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public interface Blas {

  /**
   * Assigns the result of a function to each cell; <tt>x[row,col] = function(x[row,col])</tt>.
   *
   * @param A        the matrix to modify.
   * @param function a function object taking as argument the current cell's value.
   * @see org.apache.mahout.math.function.Functions
   */
  void assign(DoubleMatrix2D A, UnaryFunction function);

  /**
   * Assigns the result of a function to each cell; <tt>x[row,col] = function(x[row,col],y[row,col])</tt>.
   *
   * @param x        the matrix to modify.
   * @param y        the secondary matrix to operate on.
   * @param function a function object taking as first argument the current cell's value of <tt>this</tt>, and as second
   *                 argument the current cell's value of <tt>y</tt>,
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>x.columns() != y.columns() || x.rows() != y.rows()</tt>
   * @see org.apache.mahout.math.function.Functions
   */
  void assign(DoubleMatrix2D x, DoubleMatrix2D y, BinaryFunction function);

  /**
   * Returns the sum of absolute values; <tt>|x[0]| + |x[1]| + ... </tt>. In fact equivalent to
   * <tt>x.aggregate(Functions.plus, org.apache.mahout.math.function.Functions.abs)</tt>.
   *
   * @param x the first vector.
   */
  double dasum(DoubleMatrix1D x);

  /**
   * Combined vector scaling; <tt>y = y + alpha*x</tt>. In fact equivalent to <tt>y.assign(x,org.apache.mahout.math.function.Functions.plusMult(alpha))</tt>.
   *
   * @param alpha a scale factor.
   * @param x     the first source vector.
   * @param y     the second source vector, this is also the vector where results are stored.
   * @throws IllegalArgumentException <tt>x.size() != y.size()</tt>..
   */
  void daxpy(double alpha, DoubleMatrix1D x, DoubleMatrix1D y);

  /**
   * Combined matrix scaling; <tt>B = B + alpha*A</tt>. In fact equivalent to <tt>B.assign(A,org.apache.mahout.math.function.Functions.plusMult(alpha))</tt>.
   *
   * @param alpha a scale factor.
   * @param A     the first source matrix.
   * @param B     the second source matrix, this is also the matrix where results are stored.
   * @throws IllegalArgumentException if <tt>A.columns() != B.columns() || A.rows() != B.rows()</tt>.
   */
  void daxpy(double alpha, DoubleMatrix2D A, DoubleMatrix2D B);

  /**
   * Vector assignment (copying); <tt>y = x</tt>. In fact equivalent to <tt>y.assign(x)</tt>.
   *
   * @param x the source vector.
   * @param y the destination vector.
   * @throws IllegalArgumentException <tt>x.size() != y.size()</tt>.
   */
  void dcopy(DoubleMatrix1D x, DoubleMatrix1D y);

  /**
   * Matrix assignment (copying); <tt>B = A</tt>. In fact equivalent to <tt>B.assign(A)</tt>.
   *
   * @param A the source matrix.
   * @param B the destination matrix.
   * @throws IllegalArgumentException if <tt>A.columns() != B.columns() || A.rows() != B.rows()</tt>.
   */
  void dcopy(DoubleMatrix2D A, DoubleMatrix2D B);

  /**
   * Returns the dot product of two vectors x and y, which is <tt>Sum(x[i]*y[i])</tt>. In fact equivalent to
   * <tt>x.zDotProduct(y)</tt>.
   *
   * @param x the first vector.
   * @param y the second vector.
   * @return the sum of products.
   * @throws IllegalArgumentException if <tt>x.size() != y.size()</tt>.
   */
  double ddot(DoubleMatrix1D x, DoubleMatrix1D y);

  /**
   * Generalized linear algebraic matrix-matrix multiply; <tt>C = alpha*A*B + beta*C</tt>. In fact equivalent to
   * <tt>A.zMult(B,C,alpha,beta,transposeA,transposeB)</tt>. Note: Matrix shape conformance is checked <i>after</i>
   * potential transpositions.
   *
   * @param transposeA set this flag to indicate that the multiplication shall be performed on A'.
   * @param transposeB set this flag to indicate that the multiplication shall be performed on B'.
   * @param alpha      a scale factor.
   * @param A          the first source matrix.
   * @param B          the second source matrix.
   * @param beta       a scale factor.
   * @param C          the third source matrix, this is also the matrix where results are stored.
   * @throws IllegalArgumentException if <tt>B.rows() != A.columns()</tt>.
   * @throws IllegalArgumentException if <tt>C.rows() != A.rows() || C.columns() != B.columns()</tt>.
   * @throws IllegalArgumentException if <tt>A == C || B == C</tt>.
   */
  void dgemm(boolean transposeA, boolean transposeB, double alpha, DoubleMatrix2D A, DoubleMatrix2D B, double beta,
             DoubleMatrix2D C);

  /**
   * Generalized linear algebraic matrix-vector multiply; <tt>y = alpha*A*x + beta*y</tt>. In fact equivalent to
   * <tt>A.zMult(x,y,alpha,beta,transposeA)</tt>. Note: Matrix shape conformance is checked <i>after</i> potential
   * transpositions.
   *
   * @param transposeA set this flag to indicate that the multiplication shall be performed on A'.
   * @param alpha      a scale factor.
   * @param A          the source matrix.
   * @param x          the first source vector.
   * @param beta       a scale factor.
   * @param y          the second source vector, this is also the vector where results are stored.
   * @throws IllegalArgumentException <tt>A.columns() != x.size() || A.rows() != y.size())</tt>..
   */
  void dgemv(boolean transposeA, double alpha, DoubleMatrix2D A, DoubleMatrix1D x, double beta, DoubleMatrix1D y);

  /**
   * Performs a rank 1 update; <tt>A = A + alpha*x*y'</tt>. Example:
   * <pre>
   * A = { {6,5}, {7,6} }, x = {1,2}, y = {3,4}, alpha = 1 -->
   * A = { {9,9}, {13,14} }
   * </pre>
   *
   * @param alpha a scalar.
   * @param x     an m element vector.
   * @param y     an n element vector.
   * @param A     an m by n matrix.
   */
  void dger(double alpha, DoubleMatrix1D x, DoubleMatrix1D y, DoubleMatrix2D A);

  /**
   * Return the 2-norm; <tt>sqrt(x[0]^2 + x[1]^2 + ...)</tt>. In fact equivalent to
   * <tt>Math.sqrt(Algebra.DEFAULT.norm2(x))</tt>.
   *
   * @param x the vector.
   */
  double dnrm2(DoubleMatrix1D x);

  /**
   * Applies a givens plane rotation to (x,y); <tt>x = c*x + s*y; y = c*y - s*x</tt>.
   *
   * @param x the first vector.
   * @param y the second vector.
   * @param c the cosine of the angle of rotation.
   * @param s the sine of the angle of rotation.
   */
  void drot(DoubleMatrix1D x, DoubleMatrix1D y, double c, double s);

  /**
   * Constructs a Givens plane rotation for <tt>(a,b)</tt>. Taken from the LINPACK translation from FORTRAN to Java,
   * interface slightly modified. In the LINPACK listing DROTG is attributed to Jack Dongarra
   *
   * @param a      rotational elimination parameter a.
   * @param b      rotational elimination parameter b.
   * @param rotvec Must be at least of length 4. On output contains the values <tt>{a,b,c,s}</tt>.
   */
  void drotg(double a, double b, double[] rotvec);

  /**
   * Vector scaling; <tt>x = alpha*x</tt>. In fact equivalent to <tt>x.assign(Functions.mult(alpha))</tt>.
   *
   * @param alpha a scale factor.
   * @param x     the first vector.
   */
  void dscal(double alpha, DoubleMatrix1D x);

  /**
   * Matrix scaling; <tt>A = alpha*A</tt>. In fact equivalent to <tt>A.assign(Functions.mult(alpha))</tt>.
   *
   * @param alpha a scale factor.
   * @param A     the matrix.
   */
  void dscal(double alpha, DoubleMatrix2D A);

  /**
   * Swaps the elements of two vectors; <tt>y <==> x</tt>. In fact equivalent to <tt>y.swap(x)</tt>.
   *
   * @param x the first vector.
   * @param y the second vector.
   * @throws IllegalArgumentException <tt>x.size() != y.size()</tt>.
   */
  void dswap(DoubleMatrix1D x, DoubleMatrix1D y);

  /**
   * Swaps the elements of two matrices; <tt>B <==> A</tt>.
   *
   * @param x the first matrix.
   * @param y the second matrix.
   * @throws IllegalArgumentException if <tt>A.columns() != B.columns() || A.rows() != B.rows()</tt>.
   */
  void dswap(DoubleMatrix2D x, DoubleMatrix2D y);

  /**
   * Symmetric matrix-vector multiplication; <tt>y = alpha*A*x + beta*y</tt>. Where alpha and beta are scalars, x and y
   * are n element vectors and A is an n by n symmetric matrix. A can be in upper or lower triangular format.
   *
   * @param isUpperTriangular is A upper triangular or lower triangular part to be used?
   * @param alpha             scaling factor.
   * @param A                 the source matrix.
   * @param x                 the first source vector.
   * @param beta              scaling factor.
   * @param y                 the second vector holding source and destination.
   */
  void dsymv(boolean isUpperTriangular, double alpha, DoubleMatrix2D A, DoubleMatrix1D x, double beta,
             DoubleMatrix1D y);

  /**
   * Triangular matrix-vector multiplication; <tt>x = A*x</tt> or <tt>x = A'*x</tt>. Where x is an n element vector and
   * A is an n by n unit, or non-unit, upper or lower triangular matrix.
   *
   * @param isUpperTriangular is A upper triangular or lower triangular?
   * @param transposeA        set this flag to indicate that the multiplication shall be performed on A'.
   * @param isUnitTriangular  true --> A is assumed to be unit triangular; false --> A is not assumed to be unit
   *                          triangular
   * @param A                 the source matrix.
   * @param x                 the vector holding source and destination.
   */
  void dtrmv(boolean isUpperTriangular, boolean transposeA, boolean isUnitTriangular, DoubleMatrix2D A,
             DoubleMatrix1D x);

  /**
   * Returns the index of largest absolute value; <tt>i such that |x[i]| == max(|x[0]|,|x[1]|,...).</tt>.
   *
   * @param x the vector to search through.
   * @return the index of largest absolute value (-1 if x is empty).
   */
  int idamax(DoubleMatrix1D x);


}
