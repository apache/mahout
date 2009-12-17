/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.list.ObjectArrayList;
import org.apache.mahout.math.matrix.impl.DenseObjectMatrix1D;
import org.apache.mahout.math.matrix.impl.SparseObjectMatrix1D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class ObjectFactory1D extends PersistentObject {

  /** A factory producing dense matrices. */
  public static final ObjectFactory1D dense = new ObjectFactory1D();

  /** A factory producing sparse matrices. */
  public static final ObjectFactory1D sparse = new ObjectFactory1D();

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected ObjectFactory1D() {
  }

  /**
   * C = A||B; Constructs a new matrix which is the concatenation of two other matrices. Example: <tt>0 1</tt> append
   * <tt>3 4</tt> --> <tt>0 1 3 4</tt>.
   */
  public ObjectMatrix1D append(ObjectMatrix1D A, ObjectMatrix1D B) {
    // concatenate
    ObjectMatrix1D matrix = make(A.size() + B.size());
    matrix.viewPart(0, A.size()).assign(A);
    matrix.viewPart(A.size(), B.size()).assign(B);
    return matrix;
  }

  /** Constructs a matrix which is the concatenation of all given parts. Cells are copied. */
  public ObjectMatrix1D make(ObjectMatrix1D[] parts) {
    if (parts.length == 0) {
      return make(0);
    }

    int size = 0;
    for (ObjectMatrix1D part1 : parts) {
      size += part1.size();
    }

    ObjectMatrix1D vector = make(size);
    size = 0;
    for (ObjectMatrix1D part : parts) {
      vector.viewPart(size, part.size()).assign(part);
      size += part.size();
    }

    return vector;
  }

  /**
   * Constructs a matrix with the given cell values. The values are copied. So subsequent changes in <tt>values</tt> are
   * not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   */
  public ObjectMatrix1D make(Object[] values) {
    if (this == sparse) {
      return new SparseObjectMatrix1D(values);
    } else {
      return new DenseObjectMatrix1D(values);
    }
  }

  /** Constructs a matrix with the given shape, each cell initialized with zero. */
  public ObjectMatrix1D make(int size) {
    if (this == sparse) {
      return new SparseObjectMatrix1D(size);
    }
    return new DenseObjectMatrix1D(size);
  }

  /** Constructs a matrix with the given shape, each cell initialized with the given value. */
  public ObjectMatrix1D make(int size, Object initialValue) {
    return make(size).assign(initialValue);
  }

  /**
   * Constructs a matrix from the values of the given list. The values are copied. So subsequent changes in
   * <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   * @return a new matrix.
   */
  public ObjectMatrix1D make(ObjectArrayList values) {
    int size = values.size();
    ObjectMatrix1D vector = make(size);
    for (int i = size; --i >= 0;) {
      vector.set(i, values.get(i));
    }
    return vector;
  }

  /**
   * C = A||A||..||A; Constructs a new matrix which is concatenated <tt>repeat</tt> times. Example:
   * <pre>
   * 0 1
   * repeat(3) -->
   * 0 1 0 1 0 1
   * </pre>
   */
  public ObjectMatrix1D repeat(ObjectMatrix1D A, int repeat) {
    int size = A.size();
    ObjectMatrix1D matrix = make(repeat * size);
    for (int i = repeat; --i >= 0;) {
      matrix.viewPart(size * i, size).assign(A);
    }
    return matrix;
  }

  /**
   * Constructs a list from the given matrix. The values are copied. So subsequent changes in <tt>values</tt> are not
   * reflected in the list, and vice-versa.
   *
   * @param values The values to be filled into the new list.
   * @return a new list.
   */
  public org.apache.mahout.math.list.ObjectArrayList toList(ObjectMatrix1D values) {
    int size = values.size();
    ObjectArrayList list = new ObjectArrayList(size);
    list.setSize(size);
    for (int i = size; --i >= 0;) {
      list.set(i, values.get(i));
    }
    return list;
  }
}
