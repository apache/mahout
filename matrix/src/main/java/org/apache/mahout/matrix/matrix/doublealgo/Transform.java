/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.doublealgo;

import org.apache.mahout.jet.math.Functions;
import org.apache.mahout.matrix.matrix.DoubleMatrix1D;
import org.apache.mahout.matrix.matrix.DoubleMatrix2D;
/**
 Deprecated; Basic element-by-element transformations on {@link org.apache.mahout.matrix.matrix.DoubleMatrix1D} and {@link org.apache.mahout.matrix.matrix.DoubleMatrix2D}.
 All transformations modify the first argument matrix to hold the result of the transformation.
 Use idioms like <tt>result = mult(matrix.copy(),5)</tt> to leave source matrices unaffected.
 <p>
 If your favourite transformation is not provided by this class, consider using method <tt>assign</tt> in combination with prefabricated function objects of {@link org.apache.mahout.jet.math.Functions},
 using idioms like
 <table>
 <td class="PRE">
 <pre>
 org.apache.mahout.jet.math.Functions F = org.apache.mahout.jet.math.Functions.functions; // alias
 matrix.assign(F.square);
 matrix.assign(F.sqrt);
 matrix.assign(F.sin);
 matrix.assign(F.log);
 matrix.assign(F.log(b));
 matrix.assign(otherMatrix, F.min);
 matrix.assign(otherMatrix, F.max);
 </pre>
 </td>
 </table>
 Here are some <a href="../doc-files/functionObjects.html">other examples</a>.
 <p>
 Implementation: Performance optimized for medium to very large matrices.
 In fact, there is now nomore a performance advantage in using this class; The assign (transform) methods directly defined on matrices are now just as fast.
 Thus, this class will soon be removed altogether.

 @deprecated
 @author wolfgang.hoschek@cern.ch
 @version 1.0, 09/24/99
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Transform extends org.apache.mahout.matrix.PersistentObject {

  /**
   * Little trick to allow for "aliasing", that is, renaming this class. Normally you would write
   * <pre>
   * Transform.mult(myMatrix,2);
   * Transform.plus(myMatrix,5);
   * </pre>
   * Since this class has only static methods, but no instance methods you can also shorten the name "DoubleTransform"
   * to a name that better suits you, for example "Trans".
   * <pre>
   * Transform T = Transform.transform; // kind of "alias"
   * T.mult(myMatrix,2);
   * T.plus(myMatrix,5);
   * </pre>
   */
  public static final Transform transform = new Transform();

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
