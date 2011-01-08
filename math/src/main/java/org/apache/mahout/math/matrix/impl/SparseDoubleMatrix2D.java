/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.impl;

import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.PlusMult;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.IntDoubleProcedure;
import org.apache.mahout.math.function.IntIntDoubleFunction;
import org.apache.mahout.math.function.Mult;
import org.apache.mahout.math.map.AbstractIntDoubleMap;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public final class SparseDoubleMatrix2D extends DoubleMatrix2D {
  /*
   * The elements of the matrix.
   */
  final AbstractIntDoubleMap elements;

  /**
   * Constructs a matrix with a copy of the given values. <tt>values</tt> is required to have the form
   * <tt>values[row][column]</tt> and have exactly the same number of columns in every row. <p> The values are copied.
   * So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   * @throws IllegalArgumentException if <tt>for any 1 &lt;= row &lt; values.length: values[row].length !=
   *                                  values[row-1].length</tt>.
   */
  public SparseDoubleMatrix2D(double[][] values) {
    this(values.length, values.length == 0 ? 0 : values[0].length);
    assign(values);
  }

  /**
   * Constructs a matrix with a given number of rows and columns and default memory usage. All entries are initially
   * <tt>0</tt>.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>.
   */
  public SparseDoubleMatrix2D(int rows, int columns) {
    this(rows, columns, rows * (columns / 1000), 0.2, 0.5);
  }

  /**
   * Constructs a matrix with a given number of rows and columns using memory as specified. All entries are initially
   * <tt>0</tt>. For details related to memory usage see {@link org.apache.mahout.math.map.OpenIntDoubleHashMap}.
   *
   * @param rows            the number of rows the matrix shall have.
   * @param columns         the number of columns the matrix shall have.
   * @param initialCapacity the initial capacity of the hash map. If not known, set <tt>initialCapacity=0</tt> or
   *                        small.
   * @param minLoadFactor   the minimum load factor of the hash map.
   * @param maxLoadFactor   the maximum load factor of the hash map.
   * @throws IllegalArgumentException if <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) ||
   *                                  (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >=
   *                                  maxLoadFactor)</tt>.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>.
   */
  public SparseDoubleMatrix2D(int rows, int columns, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
    setUp(rows, columns);
    this.elements = new OpenIntDoubleHashMap(initialCapacity, minLoadFactor, maxLoadFactor);
  }

  /**
   * Sets all cells to the state specified by <tt>value</tt>.
   *
   * @param value the value to be filled into the cells.
   * @return <tt>this</tt> (for convenience only).
   */
  @Override
  public DoubleMatrix2D assign(double value) {
    // overriden for performance only
    if (this.isNoView && value == 0) {
      this.elements.clear();
    } else {
      super.assign(value);
    }
    return this;
  }

  /**
   * Assigns the result of a function to each cell; <tt>x[row,col] = function(x[row,col])</tt>. <p> <b>Example:</b>
   * <pre>
   * matrix = 2 x 2 matrix
   * 0.5 1.5
   * 2.5 3.5
   *
   * // change each cell to its sine
   * matrix.assign(Functions.sin);
   * -->
   * 2 x 2 matrix
   * 0.479426  0.997495
   * 0.598472 -0.350783
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.
   *
   * @param function a function object taking as argument the current cell's value.
   * @see org.apache.mahout.math.function.Functions
   */
  @Override
  public void assign(DoubleFunction function) {
    if (this.isNoView && function instanceof Mult) { // x[i] = mult*x[i]
      this.elements.assign(function);
    } else {
      super.assign(function);
    }
  }

  /**
   * Replaces all cell values of the receiver with the values of another matrix. Both matrices must have the same number
   * of rows and columns. If both matrices share the same cells (as is the case if they are views derived from the same
   * matrix) and intersect in an ambiguous way, then replaces <i>as if</i> using an intermediate auxiliary deep copy of
   * <tt>other</tt>.
   *
   * @param source the source matrix to copy from (may be identical to the receiver).
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>columns() != source.columns() || rows() != source.rows()</tt>
   */
  @Override
  public DoubleMatrix2D assign(DoubleMatrix2D source) {
    // overriden for performance only
    if (!(source instanceof SparseDoubleMatrix2D)) {
      return super.assign(source);
    }
    SparseDoubleMatrix2D other = (SparseDoubleMatrix2D) source;
    if (other == this) {
      return this;
    } // nothing to do
    checkShape(other);

    if (this.isNoView && other.isNoView) { // quickest
      this.elements.assign(other.elements);
      return this;
    }
    return super.assign(source);
  }

  @Override
  public DoubleMatrix2D assign(final DoubleMatrix2D y,
                               DoubleDoubleFunction function) {
    if (!this.isNoView) {
      return super.assign(y, function);
    }

    checkShape(y);

    if (function instanceof PlusMult) { // x[i] = x[i] + alpha*y[i]
      final double alpha = ((PlusMult) function).getMultiplicator();
      if (alpha == 0) {
        return this;
      } // nothing to do
      y.forEachNonZero(
          new IntIntDoubleFunction() {
            public double apply(int i, int j, double value) {
              setQuick(i, j, getQuick(i, j) + alpha * value);
              return value;
            }
          }
      );
      return this;
    }

    if (function == Functions.MULT) { // x[i] = x[i] * y[i]
      this.elements.forEachPair(
          new IntDoubleProcedure() {
            public boolean apply(int key, double value) {
              int i = key / columns;
              int j = key % columns;
              double r = value * y.getQuick(i, j);
              if (r != value) {
                elements.put(key, r);
              }
              return true;
            }
          }
      );
    }

    if (function == Functions.DIV) { // x[i] = x[i] / y[i]
      this.elements.forEachPair(
          new IntDoubleProcedure() {
            public boolean apply(int key, double value) {
              int i = key / columns;
              int j = key % columns;
              double r = value / y.getQuick(i, j);
              if (r != value) {
                elements.put(key, r);
              }
              return true;
            }
          }
      );
    }

    return super.assign(y, function);
  }

  /** Returns the number of cells having non-zero values. */
  @Override
  public int cardinality() {
    return this.isNoView ? this.elements.size() : super.cardinality();
  }

  /**
   * Ensures that the receiver can hold at least the specified number of non-zero cells without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver. <p> This
   * method never need be called; it is for performance tuning only. Calling this method before tt>set()</tt>ing a large
   * number of non-zero values boosts performance, because the receiver will grow only once instead of potentially many
   * times and hash collisions get less probable.
   *
   * @param minCapacity the desired minimum number of non-zero cells.
   */
  @Override
  public void ensureCapacity(int minCapacity) {
    this.elements.ensureCapacity(minCapacity);
  }

  @Override
  public void forEachNonZero(final org.apache.mahout.math.function.IntIntDoubleFunction function) {
    if (this.isNoView) {
      this.elements.forEachPair(
          new IntDoubleProcedure() {
            public boolean apply(int key, double value) {
              int i = key / columns;
              int j = key % columns;
              double r = function.apply(i, j, value);
              if (r != value) {
                elements.put(key, r);
              }
              return true;
            }
          }
      );
    } else {
      super.forEachNonZero(function);
    }
  }

  /**
   * Returns the matrix cell value at coordinate <tt>[row,column]</tt>.
   *
   * <p>Provided with invalid parameters this method may return invalid objects without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the coordinate is within bounds.</b> Precondition
   * (unchecked): <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
   *
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @return the value at the specified coordinate.
   */
  @Override
  public double getQuick(int row, int column) {
    //if (debug) if (column<0 || column>=columns || row<0 || row>=rows)
    // throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
    //return this.elements.get(index(row,column));
    //manually inlined:
    return this.elements.get(rowZero + row * rowStride + columnZero + column * columnStride);
  }

  /**
   * Returns <tt>true</tt> if both matrices share common cells. More formally, returns <tt>true</tt> if at least one of
   * the following conditions is met <ul> <li>the receiver is a view of the other matrix <li>the other matrix is a view
   * of the receiver <li><tt>this == other</tt> </ul>
   */
  @Override
  protected boolean haveSharedCellsRaw(DoubleMatrix2D other) {
    if (other instanceof SelectedSparseDoubleMatrix2D) {
      SelectedSparseDoubleMatrix2D otherMatrix = (SelectedSparseDoubleMatrix2D) other;
      return this.elements == otherMatrix.elements;
    } else if (other instanceof SparseDoubleMatrix2D) {
      SparseDoubleMatrix2D otherMatrix = (SparseDoubleMatrix2D) other;
      return this.elements == otherMatrix.elements;
    }
    return false;
  }

  /**
   * Returns the position of the given coordinate within the (virtual or non-virtual) internal 1-dimensional array.
   *
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   */
  @Override
  protected int index(int row, int column) {
    // return super.index(row,column);
    // manually inlined for speed:
    return rowZero + row * rowStride + columnZero + column * columnStride;
  }

  /**
   * Construct and returns a new empty matrix <i>of the same dynamic type</i> as the receiver, having the specified
   * number of rows and columns. For example, if the receiver is an instance of type <tt>DenseDoubleMatrix2D</tt> the
   * new matrix must also be of type <tt>DenseDoubleMatrix2D</tt>, if the receiver is an instance of type
   * <tt>SparseDoubleMatrix2D</tt> the new matrix must also be of type <tt>SparseDoubleMatrix2D</tt>, etc. In general,
   * the new matrix should have internal parametrization as similar as possible.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @return a new empty matrix of the same dynamic type.
   */
  @Override
  public DoubleMatrix2D like(int rows, int columns) {
    return new SparseDoubleMatrix2D(rows, columns);
  }

  /**
   * Construct and returns a new 1-d matrix <i>of the corresponding dynamic type</i>, entirelly independent of the
   * receiver. For example, if the receiver is an instance of type <tt>DenseDoubleMatrix2D</tt> the new matrix must be
   * of type <tt>DenseDoubleMatrix1D</tt>, if the receiver is an instance of type <tt>SparseDoubleMatrix2D</tt> the new
   * matrix must be of type <tt>SparseDoubleMatrix1D</tt>, etc.
   *
   * @param size the number of cells the matrix shall have.
   * @return a new matrix of the corresponding dynamic type.
   */
  @Override
  public DoubleMatrix1D like1D(int size) {
    return new SparseDoubleMatrix1D(size);
  }

  /**
   * Construct and returns a new 1-d matrix <i>of the corresponding dynamic type</i>, sharing the same cells. For
   * example, if the receiver is an instance of type <tt>DenseDoubleMatrix2D</tt> the new matrix must be of type
   * <tt>DenseDoubleMatrix1D</tt>, if the receiver is an instance of type <tt>SparseDoubleMatrix2D</tt> the new matrix
   * must be of type <tt>SparseDoubleMatrix1D</tt>, etc.
   *
   * @param size   the number of cells the matrix shall have.
   * @param offset the index of the first element.
   * @param stride the number of indexes between any two elements, i.e. <tt>index(i+1)-index(i)</tt>.
   * @return a new matrix of the corresponding dynamic type.
   */
  @Override
  protected DoubleMatrix1D like1D(int size, int offset, int stride) {
    return new SparseDoubleMatrix1D(size, this.elements, offset, stride);
  }

  /**
   * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the specified value.
   *
   * <p>Provided with invalid parameters this method may access illegal indexes without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the coordinate is within bounds.</b> Precondition
   * (unchecked): <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
   *
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @param value  the value to be filled into the specified cell.
   */
  @Override
  public void setQuick(int row, int column, double value) {
    //if (debug) if (column<0 || column>=columns || row<0 || row>=rows)
    // throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
    //int index =  index(row,column);
    //manually inlined:
    int index = rowZero + row * rowStride + columnZero + column * columnStride;

    //if (value == 0 || Math.abs(value) < TOLERANCE)
    if (value == 0) {
      this.elements.removeKey(index);
    } else {
      this.elements.put(index, value);
    }
  }

  /**
   * Construct and returns a new selection view.
   *
   * @param rowOffsets    the offsets of the visible elements.
   * @param columnOffsets the offsets of the visible elements.
   * @return a new view.
   */
  @Override
  protected DoubleMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
    return new SelectedSparseDoubleMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
  }

  @Override
  public DoubleMatrix1D zMult(DoubleMatrix1D y, DoubleMatrix1D z, double alpha, double beta, final boolean transposeA) {
    int m = rows;
    int n = columns;
    if (transposeA) {
      m = columns;
      n = rows;
    }

    boolean ignore = z == null;
    if (ignore) {
      z = new DenseDoubleMatrix1D(m);
    }

    if (!(this.isNoView && y instanceof DenseDoubleMatrix1D && z instanceof DenseDoubleMatrix1D)) {
      return super.zMult(y, z, alpha, beta, transposeA);
    }

    if (n != y.size() || m > z.size()) {
      throw new IllegalArgumentException("Incompatible args");
    }

    if (!ignore) {
      z.assign(Functions.mult(beta / alpha));
    }

    DenseDoubleMatrix1D zz = (DenseDoubleMatrix1D) z;
    final double[] zElements = zz.elements;
    final int zStride = zz.stride;
    final int zi = z.index(0);

    DenseDoubleMatrix1D yy = (DenseDoubleMatrix1D) y;
    final double[] yElements = yy.elements;
    final int yStride = yy.stride;
    final int yi = y.index(0);

    if (yElements == null || zElements == null) {
      throw new IllegalStateException();
    }

    this.elements.forEachPair(
        new IntDoubleProcedure() {
          public boolean apply(int key, double value) {
            int i = key / columns;
            int j = key % columns;
            if (transposeA) {
              int tmp = i;
              i = j;
              j = tmp;
            }
            zElements[zi + zStride * i] += value * yElements[yi + yStride * j];
            return true;
          }
        }
    );

    if (alpha != 1.0) {
      z.assign(Functions.mult(alpha));
    }
    return z;
  }

  @Override
  public DoubleMatrix2D zMult(DoubleMatrix2D B, DoubleMatrix2D C, final double alpha, double beta,
                              final boolean transposeA, boolean transposeB) {
    if (!(this.isNoView)) {
      return super.zMult(B, C, alpha, beta, transposeA, transposeB);
    }
    if (transposeB) {
      B = B.viewDice();
    }
    int m = rows;
    int n = columns;
    if (transposeA) {
      m = columns;
      n = rows;
    }
    int p = B.columns;
    boolean ignore = (C == null);
    if (C == null) {
      C = new DenseDoubleMatrix2D(m, p);
    }

    if (B.rows != n) {
      throw new IllegalArgumentException("Matrix2D inner dimensions must agree");
    }
    if (C.rows != m || C.columns != p) {
      throw new IllegalArgumentException("Incompatible result matrix");
    }
    if (this == C || B == C) {
      throw new IllegalArgumentException("Matrices must not be identical");
    }

    if (!ignore) {
      C.assign(Functions.mult(beta));
    }

    // cache views
    final DoubleMatrix1D[] Brows = new DoubleMatrix1D[n];
    for (int i = n; --i >= 0;) {
      Brows[i] = B.viewRow(i);
    }
    final DoubleMatrix1D[] Crows = new DoubleMatrix1D[m];
    for (int i = m; --i >= 0;) {
      Crows[i] = C.viewRow(i);
    }

    final PlusMult fun = PlusMult.plusMult(0);

    this.elements.forEachPair(
        new IntDoubleProcedure() {
          public boolean apply(int key, double value) {
            int i = key / columns;
            int j = key % columns;
            fun.setMultiplicator(value * alpha);
            if (transposeA) {
              Crows[j].assign(Brows[i], fun);
            } else {
              Crows[i].assign(Brows[j], fun);
            }
            return true;
          }
        }
    );

    return C;
  }
}
