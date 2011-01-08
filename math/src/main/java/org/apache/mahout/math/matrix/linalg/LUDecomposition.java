/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import java.io.Serializable;

import org.apache.mahout.math.matrix.DoubleMatrix2D;

/**
 * For an <tt>m x n</tt> matrix <tt>A</tt> with <tt>m >= n</tt>, the LU decomposition is an <tt>m x n</tt> unit lower
 * triangular matrix <tt>L</tt>, an <tt>n x n</tt> upper triangular matrix <tt>U</tt>, and a permutation vector
 * <tt>piv</tt> of length <tt>m</tt> so that <tt>A(piv,:) = L*U</tt>; If <tt>m < n</tt>, then <tt>L</tt> is <tt>m x
 * m</tt> and <tt>U</tt> is <tt>m x n</tt>. <P> The LU decomposition with pivoting always exists, even if the matrix is
 * singular, so the constructor will never fail.  The primary use of the LU decomposition is in the solution of square
 * systems of simultaneous linear equations.  This will fail if <tt>isNonsingular()</tt> returns false.
 */
public final class LUDecomposition implements Serializable {

  private final LUDecompositionQuick quick;

  /**
   * Constructs and returns a new LU Decomposition object; The decomposed matrices can be retrieved via instance methods
   * of the returned decomposition object.
   *
   * @param a Rectangular matrix
   */
  public LUDecomposition(DoubleMatrix2D a) {
    quick = new LUDecompositionQuick(0); // zero tolerance for compatibility with Jama
    quick.decompose(a.copy());
  }

  /**
   * Returns the determinant, <tt>det(A)</tt>.
   *
   * @throws IllegalArgumentException Matrix must be square
   */
  public double det() {
    return quick.det();
  }

  /**
   * Solves <tt>A*X = B</tt>.
   *
   * @param b A matrix with as many rows as <tt>A</tt> and any number of columns.
   * @return <tt>X</tt> so that <tt>L*U*X = B(piv,:)</tt>.
   * @throws IllegalArgumentException if </tt>B.rows() != A.rows()</tt>.
   * @throws IllegalArgumentException if A is singular, that is, if <tt>!this.isNonsingular()</tt>.
   * @throws IllegalArgumentException if <tt>A.rows() < A.columns()</tt>.
   */

  public DoubleMatrix2D solve(DoubleMatrix2D b) {
    DoubleMatrix2D x = b.copy();
    quick.solve(x);
    return x;
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
    return quick.toString();
  }
}
