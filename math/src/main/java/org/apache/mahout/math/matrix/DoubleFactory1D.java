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
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.jet.random.engine.MersenneTwister;
import org.apache.mahout.math.jet.random.sampling.RandomSamplingAssistant;
import org.apache.mahout.math.list.AbstractDoubleList;
import org.apache.mahout.math.list.DoubleArrayList;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix1D;
import org.apache.mahout.math.matrix.impl.SparseDoubleMatrix1D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class DoubleFactory1D extends PersistentObject {

  /** A factory producing dense matrices. */
  public static final DoubleFactory1D dense = new DoubleFactory1D();

  /** A factory producing sparse matrices. */
  private static final DoubleFactory1D sparse = new DoubleFactory1D();

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected DoubleFactory1D() {
  }

  /**
   * C = A||B; Constructs a new matrix which is the concatenation of two other matrices. Example: <tt>0 1</tt> append
   * <tt>3 4</tt> --> <tt>0 1 3 4</tt>.
   */
  public DoubleMatrix1D append(DoubleMatrix1D A, DoubleMatrix1D B) {
    // concatenate
    DoubleMatrix1D matrix = make(A.size() + B.size());
    matrix.viewPart(0, A.size()).assign(A);
    matrix.viewPart(A.size(), B.size()).assign(B);
    return matrix;
  }

  /** Constructs a matrix with cells having ascending values. For debugging purposes. Example: <tt>0 1 2</tt> */
  public DoubleMatrix1D ascending(int size) {
    return descending(size).assign(Functions.chain(Functions.NEGATE, Functions.minus(size)));
  }

  /** Constructs a matrix with cells having descending values. For debugging purposes. Example: <tt>2 1 0</tt> */
  public DoubleMatrix1D descending(int size) {
    DoubleMatrix1D matrix = make(size);
    int v = 0;
    for (int i = size; --i >= 0;) {
      matrix.setQuick(i, v++);
    }
    return matrix;
  }

  /**
   * Constructs a matrix with the given cell values. The values are copied. So subsequent changes in <tt>values</tt> are
   * not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   */
  public DoubleMatrix1D make(double[] values) {
    return this == sparse ? new SparseDoubleMatrix1D(values) : new DenseDoubleMatrix1D(values);
  }

  /** Constructs a matrix which is the concatenation of all given parts. Cells are copied. */
  public DoubleMatrix1D make(DoubleMatrix1D[] parts) {
    if (parts.length == 0) {
      return make(0);
    }

    int size = 0;
    for (DoubleMatrix1D part1 : parts) {
      size += part1.size();
    }

    DoubleMatrix1D vector = make(size);
    size = 0;
    for (DoubleMatrix1D part : parts) {
      vector.viewPart(size, part.size()).assign(part);
      size += part.size();
    }

    return vector;
  }

  /** Constructs a matrix with the given shape, each cell initialized with zero. */
  public DoubleMatrix1D make(int size) {
    if (this == sparse) {
      return new SparseDoubleMatrix1D(size);
    }
    return new DenseDoubleMatrix1D(size);
  }

  /** Constructs a matrix with the given shape, each cell initialized with the given value. */
  public DoubleMatrix1D make(int size, double initialValue) {
    return make(size).assign(initialValue);
  }

  /**
   * Constructs a matrix from the values of the given list. The values are copied. So subsequent changes in
   * <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   * @return a new matrix.
   */
  public DoubleMatrix1D make(AbstractDoubleList values) {
    int size = values.size();
    DoubleMatrix1D vector = make(size);
    for (int i = size; --i >= 0;) {
      vector.set(i, values.get(i));
    }
    return vector;
  }

  /** Constructs a matrix with uniformly distributed values in <tt>(0,1)</tt> (exclusive). */
  public DoubleMatrix1D random(int size) {
    return make(size).assign(Functions.random());
  }

  /**
   * C = A||A||..||A; Constructs a new matrix which is concatenated <tt>repeat</tt> times. Example:
   * <pre>
   * 0 1
   * repeat(3) -->
   * 0 1 0 1 0 1
   * </pre>
   */
  public DoubleMatrix1D repeat(DoubleMatrix1D A, int repeat) {
    int size = A.size();
    DoubleMatrix1D matrix = make(repeat * size);
    for (int i = repeat; --i >= 0;) {
      matrix.viewPart(size * i, size).assign(A);
    }
    return matrix;
  }

  /**
   * Constructs a randomly sampled matrix with the given shape. Randomly picks exactly
   * <tt>Math.round(size*nonZeroFraction)</tt> cells and initializes them to <tt>value</tt>, all the rest will be
   * initialized to zero. Note that this is not the same as setting each cell with probability <tt>nonZeroFraction</tt>
   * to <tt>value</tt>.
   *
   * @throws IllegalArgumentException if <tt>nonZeroFraction < 0 || nonZeroFraction > 1</tt>.
   * @see org.apache.mahout.math.jet.random.sampling.RandomSampler
   */
  public DoubleMatrix1D sample(int size, double value, double nonZeroFraction) {
    double epsilon = 1.0e-9;
    if (nonZeroFraction < 0 - epsilon || nonZeroFraction > 1 + epsilon) {
      throw new IllegalArgumentException();
    }
    if (nonZeroFraction < 0) {
      nonZeroFraction = 0;
    }
    if (nonZeroFraction > 1) {
      nonZeroFraction = 1;
    }

    DoubleMatrix1D matrix = make(size);

    int n = (int) Math.round(size * nonZeroFraction);
    if (n == 0) {
      return matrix;
    }

    RandomSamplingAssistant sampler =
        new RandomSamplingAssistant(n, size,
            new MersenneTwister());
    for (int i = size; --i >= 0;) {
      if (sampler.sampleNextElement()) {
        matrix.set(i, value);
      }
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
  public org.apache.mahout.math.list.DoubleArrayList toList(DoubleMatrix1D values) {
    int size = values.size();
    DoubleArrayList list = new DoubleArrayList(size);
    list.setSize(size);
    for (int i = size; --i >= 0;) {
      list.set(i, values.get(i));
    }
    return list;
  }
}
