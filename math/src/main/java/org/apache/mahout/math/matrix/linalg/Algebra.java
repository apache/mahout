/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import org.apache.mahout.math.GenericPermuting;
import org.apache.mahout.math.Swapper;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public final class Algebra {

  /**
   * A default Algebra object; has {@link Property#DEFAULT} attached for tolerance. Allows ommiting to construct an
   * Algebra object time and again.
   *
   * Note that this Algebra object is immutable. Any attempt to assign a new Property object to it (via method
   * <tt>setProperty</tt>), or to alter the tolerance of its property object (via <tt>property().setTolerance(...)</tt>)
   * will throw an exception.
   */
  public static final Algebra DEFAULT;

  /**
   * A default Algebra object; has {@link Property#ZERO} attached for tolerance. Allows ommiting to construct an Algebra
   * object time and again.
   *
   * Note that this Algebra object is immutable. Any attempt to assign a new Property object to it (via method
   * <tt>setProperty</tt>), or to alter the tolerance of its property object (via <tt>property().setTolerance(...)</tt>)
   * will throw an exception.
   */
  private static final Algebra ZERO;

  /** The property object attached to this instance. */
  private Property property;

  static {
    // don't use new Algebra(Property.DEFAULT.tolerance()), because then property object would be mutable.
    DEFAULT = new Algebra();
    DEFAULT.property = Property.DEFAULT; // immutable property object

    ZERO = new Algebra();
    ZERO.property = Property.ZERO; // immutable property object
  }

  /** Constructs a new instance with an equality tolerance given by <tt>Property.DEFAULT.tolerance()</tt>. */
  public Algebra() {
    this(Property.DEFAULT.tolerance());
  }

  /**
   * Constructs a new instance with the given equality tolerance.
   *
   * @param tolerance the tolerance to be used for equality operations.
   */
  public Algebra(double tolerance) {
    setProperty(new Property(tolerance));
  }

  /**
   * Returns the determinant of matrix <tt>A</tt>.
   *
   * @return the determinant.
   */
  public static double det(DoubleMatrix2D A) {
    return lu(A).det();
  }

  /** Returns sqrt(a^2 + b^2) without under/overflow. */
  static double hypot(double a, double b) {
    double r;
    if (Math.abs(a) > Math.abs(b)) {
      r = b / a;
      r = Math.abs(a) * Math.sqrt(1 + r * r);
    } else if (b != 0) {
      r = a / b;
      r = Math.abs(b) * Math.sqrt(1 + r * r);
    } else {
      r = 0.0;
    }
    return r;
  }

  /** Constructs and returns the LU-decomposition of the given matrix. */
  private static LUDecomposition lu(DoubleMatrix2D matrix) {
    return new LUDecomposition(matrix);
  }

  /**
   * Modifies the given vector <tt>A</tt> such that it is permuted as specified; Useful for pivoting. Cell <tt>A[i]</tt>
   * will go into cell <tt>A[indexes[i]]</tt>. <p> <b>Example:</b>
   * <pre>
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,2,3,1] yields
   * [A,E,C,D,B]
   * In other words A[0]<--A[0], A[1]<--A[4], A[2]<--A[2], A[3]<--A[3], A[4]<--A[1].
   *
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,1,2,3] yields
   * [A,E,B,C,D]
   * In other words A[0]<--A[0], A[1]<--A[4], A[2]<--A[1], A[3]<--A[2], A[4]<--A[3].
   * </pre>
   *
   * @param A       the vector to permute.
   * @param indexes the permutation indexes, must satisfy <tt>indexes.length==A.size() && indexes[i] >= 0 && indexes[i]
   *                < A.size()</tt>;
   * @param work    the working storage, must satisfy <tt>work.length >= A.size()</tt>; set <tt>work==null</tt> if you
   *                don't care about performance.
   * @throws IndexOutOfBoundsException if <tt>indexes.length != A.size()</tt>.
   */
  public static void permute(DoubleMatrix1D A, int[] indexes, double[] work) {
    // check validity
    int size = A.size();
    if (indexes.length != size) {
      throw new IndexOutOfBoundsException("invalid permutation");
    }

    /*
    int i=size;
    int a;
    while (--i >= 0 && (a=indexes[i])==i) if (a < 0 || a >= size)
    throw new IndexOutOfBoundsException("invalid permutation");
    if (i<0) return; // nothing to permute
    */

    if (work == null || size > work.length) {
      work = A.toArray();
    } else {
      A.toArray(work);
    }
    for (int i = size; --i >= 0;) {
      A.setQuick(i, work[indexes[i]]);
    }
  }

  /**
   * Modifies the given matrix <tt>A</tt> such that it's rows are permuted as specified; Useful for pivoting. Row
   * <tt>A[i]</tt> will go into row <tt>A[indexes[i]]</tt>. <p> <b>Example:</b>
   * <pre>
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,2,3,1] yields
   * [A,E,C,D,B]
   * In other words A[0]<--A[0], A[1]<--A[4], A[2]<--A[2], A[3]<--A[3], A[4]<--A[1].
   *
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,1,2,3] yields
   * [A,E,B,C,D]
   * In other words A[0]<--A[0], A[1]<--A[4], A[2]<--A[1], A[3]<--A[2], A[4]<--A[3].
   * </pre>
   *
   * @param A       the matrix to permute.
   * @param indexes the permutation indexes, must satisfy <tt>indexes.length==A.rows() && indexes[i] >= 0 && indexes[i]
   *                < A.rows()</tt>;
   * @param work    the working storage, must satisfy <tt>work.length >= A.rows()</tt>; set <tt>work==null</tt> if you
   *                don't care about performance.
   * @throws IndexOutOfBoundsException if <tt>indexes.length != A.rows()</tt>.
   */
  public static void permuteRows(final DoubleMatrix2D A, int[] indexes, int[] work) {
    // check validity
    int size = A.rows();
    if (indexes.length != size) {
      throw new IndexOutOfBoundsException("invalid permutation");
    }

    /*
    int i=size;
    int a;
    while (--i >= 0 && (a=indexes[i])==i) if (a < 0 || a >= size)
      throw new IndexOutOfBoundsException("invalid permutation");
    if (i<0) return; // nothing to permute
    */

    int columns = A.columns();
    if (columns < size / 10) { // quicker
      double[] doubleWork = new double[size];
      for (int j = A.columns(); --j >= 0;) {
        permute(A.viewColumn(j), indexes, doubleWork);
      }
      return;
    }

    Swapper swapper = new Swapper() {
      public void swap(int a, int b) {
        A.viewRow(a).swap(A.viewRow(b));
      }
    };

    GenericPermuting.permute(indexes, swapper, work, null);
  }

  /**
   * Returns the property object attached to this Algebra, defining tolerance.
   *
   * @return the Property object.
   * @see #setProperty(Property)
   */
  public Property property() {
    return property;
  }

  /**
   * Attaches the given property object to this Algebra, defining tolerance.
   *
   * @param property the Property object to be attached.
   * @throws UnsupportedOperationException if <tt>this==DEFAULT && property!=this.property()</tt> - The DEFAULT Algebra
   *                                       object is immutable.
   * @throws UnsupportedOperationException if <tt>this==ZERO && property!=this.property()</tt> - The ZERO Algebra object
   *                                       is immutable.
   * @see #property
   */
  public void setProperty(Property property) {
    if (this == DEFAULT && property != this.property) {
      throw new IllegalArgumentException("Attempted to modify immutable object.");
    }
    if (this == ZERO && property != this.property) {
      throw new IllegalArgumentException("Attempted to modify immutable object.");
    }
    this.property = property;
  }

  /**
   * Modifies the matrix to be a lower trapezoidal matrix.
   *
   * @return <tt>A</tt> (for convenience only).
   */
  static DoubleMatrix2D trapezoidalLower(DoubleMatrix2D A) {
    int rows = A.rows();
    int columns = A.columns();
    for (int r = rows; --r >= 0;) {
      for (int c = columns; --c >= 0;) {
        if (r < c) {
          A.setQuick(r, c, 0);
        }
      }
    }
    return A;
  }

}
