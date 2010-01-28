/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.doublealgo;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Transform extends PersistentObject {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected Transform() {
  }

  /**
   * <tt>A[i] = Math.abs(A[i])</tt>.
   *
   * @param A the matrix to modify.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D abs(DoubleMatrix1D A) {
    return A.assign(Functions.abs);
  }

  /**
   * <tt>A[row,col] = Math.abs(A[row,col])</tt>.
   *
   * @param A the matrix to modify.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D abs(DoubleMatrix2D A) {
    return A.assign(Functions.abs);
  }

  /**
   * <tt>A = A / s <=> A[i] = A[i] / s</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D div(DoubleMatrix1D A, double s) {
    return A.assign(Functions.div(s));
  }

  /**
   * <tt>A = A / B <=> A[i] = A[i] / B[i]</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D div(DoubleMatrix1D A, DoubleMatrix1D B) {
    return A.assign(B, Functions.div);
  }

  /**
   * <tt>A = A / s <=> A[row,col] = A[row,col] / s</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D div(DoubleMatrix2D A, double s) {
    return A.assign(Functions.div(s));
  }

  /**
   * <tt>A = A / B <=> A[row,col] = A[row,col] / B[row,col]</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D div(DoubleMatrix2D A, DoubleMatrix2D B) {
    return A.assign(B, Functions.div);
  }

  /**
   * <tt>A[row,col] = A[row,col] == s ? 1 : 0</tt>; ignores tolerance.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D equals(DoubleMatrix2D A, double s) {
    return A.assign(Functions.equals(s));
  }

  /**
   * <tt>A[row,col] = A[row,col] == B[row,col] ? 1 : 0</tt>; ignores tolerance.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D equals(DoubleMatrix2D A, DoubleMatrix2D B) {
    return A.assign(B, Functions.equals);
  }

  /**
   * <tt>A[row,col] = A[row,col] > s ? 1 : 0</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D greater(DoubleMatrix2D A, double s) {
    return A.assign(Functions.greater(s));
  }

  /**
   * <tt>A[row,col] = A[row,col] > B[row,col] ? 1 : 0</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D greater(DoubleMatrix2D A, DoubleMatrix2D B) {
    return A.assign(B, Functions.greater);
  }

  /**
   * <tt>A[row,col] = A[row,col] < s ? 1 : 0</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D less(DoubleMatrix2D A, double s) {
    return A.assign(Functions.less(s));
  }

  /**
   * <tt>A[row,col] = A[row,col] < B[row,col] ? 1 : 0</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D less(DoubleMatrix2D A, DoubleMatrix2D B) {
    return A.assign(B, Functions.less);
  }

  /**
   * <tt>A = A - s <=> A[i] = A[i] - s</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D minus(DoubleMatrix1D A, double s) {
    return A.assign(Functions.minus(s));
  }

  /**
   * <tt>A = A - B <=> A[i] = A[i] - B[i]</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D minus(DoubleMatrix1D A, DoubleMatrix1D B) {
    return A.assign(B, Functions.minus);
  }

  /**
   * <tt>A = A - s <=> A[row,col] = A[row,col] - s</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D minus(DoubleMatrix2D A, double s) {
    return A.assign(Functions.minus(s));
  }

  /**
   * <tt>A = A - B <=> A[row,col] = A[row,col] - B[row,col]</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D minus(DoubleMatrix2D A, DoubleMatrix2D B) {
    return A.assign(B, Functions.minus);
  }

  /**
   * <tt>A = A - B*s <=> A[i] = A[i] - B[i]*s</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D minusMult(DoubleMatrix1D A, DoubleMatrix1D B, double s) {
    return A.assign(B, Functions.minusMult(s));
  }

  /**
   * <tt>A = A - B*s <=> A[row,col] = A[row,col] - B[row,col]*s</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D minusMult(DoubleMatrix2D A, DoubleMatrix2D B, double s) {
    return A.assign(B, Functions.minusMult(s));
  }

  /**
   * <tt>A = A * s <=> A[i] = A[i] * s</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D mult(DoubleMatrix1D A, double s) {
    return A.assign(Functions.mult(s));
  }

  /**
   * <tt>A = A * B <=> A[i] = A[i] * B[i]</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D mult(DoubleMatrix1D A, DoubleMatrix1D B) {
    return A.assign(B, Functions.mult);
  }

  /**
   * <tt>A = A * s <=> A[row,col] = A[row,col] * s</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D mult(DoubleMatrix2D A, double s) {
    return A.assign(Functions.mult(s));
  }

  /**
   * <tt>A = A * B <=> A[row,col] = A[row,col] * B[row,col]</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D mult(DoubleMatrix2D A, DoubleMatrix2D B) {
    return A.assign(B, Functions.mult);
  }

  /**
   * <tt>A = -A <=> A[i] = -A[i]</tt> for all cells.
   *
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D negate(DoubleMatrix1D A) {
    return A.assign(Functions.mult(-1));
  }

  /**
   * <tt>A = -A <=> A[row,col] = -A[row,col]</tt>.
   *
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D negate(DoubleMatrix2D A) {
    return A.assign(Functions.mult(-1));
  }

  /**
   * <tt>A = A + s <=> A[i] = A[i] + s</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D plus(DoubleMatrix1D A, double s) {
    return A.assign(Functions.plus(s));
  }

  /**
   * <tt>A = A + B <=> A[i] = A[i] + B[i]</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D plus(DoubleMatrix1D A, DoubleMatrix1D B) {
    return A.assign(B, Functions.plus);
  }

  /**
   * <tt>A = A + s <=> A[row,col] = A[row,col] + s</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D plus(DoubleMatrix2D A, double s) {
    return A.assign(Functions.plus(s));
  }

  /**
   * <tt>A = A + B <=> A[row,col] = A[row,col] + B[row,col]</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D plus(DoubleMatrix2D A, DoubleMatrix2D B) {
    return A.assign(B, Functions.plus);
  }

  /**
   * <tt>A = A + B*s<=> A[i] = A[i] + B[i]*s</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D plusMult(DoubleMatrix1D A, DoubleMatrix1D B, double s) {
    return A.assign(B, Functions.plusMult(s));
  }

  /**
   * <tt>A = A + B*s <=> A[row,col] = A[row,col] + B[row,col]*s</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D plusMult(DoubleMatrix2D A, DoubleMatrix2D B, double s) {
    return A.assign(B, Functions.plusMult(s));
  }

  /**
   * <tt>A = A<sup>s</sup> <=> A[i] = Math.pow(A[i], s)</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D pow(DoubleMatrix1D A, double s) {
    return A.assign(Functions.pow(s));
  }

  /**
   * <tt>A = A<sup>B</sup> <=> A[i] = Math.pow(A[i], B[i])</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix1D pow(DoubleMatrix1D A, DoubleMatrix1D B) {
    return A.assign(B, Functions.pow);
  }

  /**
   * <tt>A = A<sup>s</sup> <=> A[row,col] = Math.pow(A[row,col], s)</tt>.
   *
   * @param A the matrix to modify.
   * @param s the scalar; can have any value.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D pow(DoubleMatrix2D A, double s) {
    return A.assign(Functions.pow(s));
  }

  /**
   * <tt>A = A<sup>B</sup> <=> A[row,col] = Math.pow(A[row,col], B[row,col])</tt>.
   *
   * @param A the matrix to modify.
   * @param B the matrix to stay unaffected.
   * @return <tt>A</tt> (for convenience only).
   */
  public static DoubleMatrix2D pow(DoubleMatrix2D A, DoubleMatrix2D B) {
    return A.assign(B, Functions.pow);
  }
}
