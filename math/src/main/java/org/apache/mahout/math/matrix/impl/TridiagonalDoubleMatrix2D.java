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
import org.apache.mahout.math.function.IntIntDoubleFunction;
import org.apache.mahout.math.jet.math.Functions;
import org.apache.mahout.math.jet.math.Mult;
import org.apache.mahout.math.jet.math.PlusMult;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/**
 * Tridiagonal 2-d matrix holding <tt>double</tt> elements. First see the <a href="package-summary.html">package
 * summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture. <p>
 * <b>Implementation:</b> TODO.
 *
 * @author wolfgang.hoschek@cern.ch
 * @version 0.9, 04/14/2000
 */
class TridiagonalDoubleMatrix2D extends WrapperDoubleMatrix2D {
  /*
   * The non zero elements of the matrix: {lower, diagonal, upper}.
   */
  private final double[] values;

  /*
  * The startIndexes and number of non zeros: {lowerStart, diagonalStart, upperStart, values.length, lowerNonZeros, diagonalNonZeros, upperNonZeros}.
  * lowerStart = 0
  * diagonalStart = lowerStart + lower.length
  * upperStart = diagonalStart + diagonal.length
  */
  private final int[] dims;

  private static final int NONZERO = 4;

  //protected double diagonal[];
  //protected double lower[];
  //protected double upper[];

  //protected int diagonalNonZeros;
  //protected int lowerNonZeros;
  //protected int upperNonZeros;
  //protected int N;

  /**
   * Constructs a matrix with a copy of the given values. <tt>values</tt> is required to have the form
   * <tt>values[row][column]</tt> and have exactly the same number of columns in every row. <p> The values are copied.
   * So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   * @throws IllegalArgumentException if <tt>for any 1 &lt;= row &lt; values.length: values[row].length !=
   *                                  values[row-1].length</tt>.
   */
  TridiagonalDoubleMatrix2D(double[][] values) {
    this(values.length, values.length == 0 ? 0 : values[0].length);
    assign(values);
  }

  /**
   * Constructs a matrix with a given number of rows and columns. All entries are initially <tt>0</tt>.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>.
   */
  TridiagonalDoubleMatrix2D(int rows, int columns) {
    super(null);
    setUp(rows, columns);

    int d = Math.min(rows, columns);
    int u = d - 1;
    int l = d - 1;
    if (rows > columns) {
      l++;
    }
    if (rows < columns) {
      u++;
    }

    values = new double[l + d + u]; // {lower, diagonal, upper}
    dims = new int[]{0, l, l + d, l + d + u, 0, 0, 0};
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
    if (value == 0) {
      for (int i = values.length; --i >= 0;) {
        values[i] = 0;
      }
      for (int i = dims.length; --i >= NONZERO;) {
        dims[i] = 0;
      }

    } else {
      super.assign(value);
    }
    return this;
  }

  @Override
  public DoubleMatrix2D assign(final org.apache.mahout.math.function.DoubleFunction function) {
    if (function instanceof Mult) { // x[i] = mult*x[i]
      double alpha = ((Mult) function).getMultiplicator();
      if (alpha == 1) {
        return this;
      }
      if (alpha == 0) {
        return assign(0);
      }
      if (alpha != alpha) {
        return assign(alpha);
      } // the funny definition of isNaN(). This should better not happen.

      forEachNonZero(
          new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int j, double value) {
              return function.apply(value);
            }
          }
      );
    } else {
      super.assign(function);
    }
    return this;
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
    if (source == this) {
      return this;
    } // nothing to do
    checkShape(source);

    if (source instanceof TridiagonalDoubleMatrix2D) {
      // quickest
      TridiagonalDoubleMatrix2D other = (TridiagonalDoubleMatrix2D) source;

      System.arraycopy(other.values, 0, this.values, 0, this.values.length);
      System.arraycopy(other.dims, 0, this.dims, 0, this.dims.length);
      return this;
    }

    if (source instanceof RCDoubleMatrix2D || source instanceof SparseDoubleMatrix2D) {
      assign(0);
      source.forEachNonZero(
          new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int j, double value) {
              setQuick(i, j, value);
              return value;
            }
          }
      );
      return this;
    }

    return super.assign(source);
  }

  @Override
  public DoubleMatrix2D assign(final DoubleMatrix2D y,
                               DoubleDoubleFunction function) {
    checkShape(y);

    if (function instanceof PlusMult) { // x[i] = x[i] + alpha*y[i]
      final double alpha = ((PlusMult) function).getMultiplicator();
      if (alpha == 0) {
        return this;
      } // nothing to do
      y.forEachNonZero(
          new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int j, double value) {
              setQuick(i, j, getQuick(i, j) + alpha * value);
              return value;
            }
          }
      );
      return this;
    }

    if (function == org.apache.mahout.math.jet.math.Functions.mult) { // x[i] = x[i] * y[i]
      forEachNonZero(
          new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int j, double value) {
              setQuick(i, j, getQuick(i, j) * y.getQuick(i, j));
              return value;
            }
          }
      );
      return this;
    }

    if (function == org.apache.mahout.math.jet.math.Functions.div) { // x[i] = x[i] / y[i]
      forEachNonZero(
          new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int j, double value) {
              setQuick(i, j, getQuick(i, j) / y.getQuick(i, j));
              return value;
            }
          }
      );
      return this;
    }

    return super.assign(y, function);
  }

  @Override
  public DoubleMatrix2D forEachNonZero(IntIntDoubleFunction function) {
    for (int kind = 0; kind <= 2; kind++) {
      int i = 0, j = 0;
      switch (kind) {
        case 0: {
          i = 1;
        } // lower
        // case 1: {   } // diagonal
        case 2: {
          j = 1;
        } // upper
      }
      int low = dims[kind];
      int high = dims[kind + 1];

      for (int k = low; k < high; k++, i++, j++) {
        double value = values[k];
        if (value != 0) {
          double r = function.apply(i, j, value);
          if (r != value) {
            if (r == 0) {
              dims[kind + NONZERO]++;
            } // one non zero more
            values[k] = r;
          }
        }
      }
    }
    return this;
  }

  /**
   * Returns the content of this matrix if it is a wrapper; or <tt>this</tt> otherwise. Override this method in
   * wrappers.
   */
  @Override
  protected DoubleMatrix2D getContent() {
    return this;
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
    int i = row;
    int j = column;

    int k = j - i + 1;
    int q = i;
    if (k == 0) {
      q = j;
    } // lower diagonal

    if (k >= 0 && k <= 2) {
      return values[dims[k] + q];
    }
    return 0;


    //int k = -1;
    //int q = 0;

    //if (i==j) { k=0; q=i; }
    //if (i==j+1) { k=1; q=j; }
    //if (i==j-1) { k=2; q=i; }

    //if (k<0) return 0;
    //return values[dims[k]+q];


    //if (i==j) return diagonal[i];
    //if (i==j+1) return lower[j];
    //if (i==j-1) return upper[i];

    //return 0;
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
    return new TridiagonalDoubleMatrix2D(rows, columns);
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
    int i = row;
    int j = column;

    boolean isZero = (value == 0);

    int k = j - i + 1;
    int q = i;
    if (k == 0) {
      q = j;
    } // lower diagonal

    if (k >= 0 && k <= 2) {
      int index = dims[k] + q;
      if (values[index] != 0) {
        if (isZero) {
          dims[k + NONZERO]--;
        } // one nonZero less
      } else {
        if (!isZero) {
          dims[k + NONZERO]++;
        } // one nonZero more
      }
      values[index] = value;
      return;
    }

    if (!isZero) {
      throw new IllegalArgumentException(
          "Can't store non-zero value to non-tridiagonal coordinate: row=" + row + ", column=" + column + ", value=" +
              value);
    }

  }

  @Override
  public DoubleMatrix1D zMult(DoubleMatrix1D y, DoubleMatrix1D z, double alpha, double beta, final boolean transposeA) {
    int m = rows;
    int n = columns;
    if (transposeA) {
      m = columns;
      n = rows;
    }

    boolean ignore = (z == null);
    if (z == null) {
      z = new DenseDoubleMatrix1D(m);
    }

    if (!(this.isNoView && y instanceof DenseDoubleMatrix1D && z instanceof DenseDoubleMatrix1D)) {
      return super.zMult(y, z, alpha, beta, transposeA);
    }

    if (n != y.size() || m > z.size()) {
      throw new IllegalArgumentException(
          "Incompatible args: " + ((transposeA ? viewDice() : this).toStringShort()) + ", " + y.toStringShort() + ", " +
              z.toStringShort());
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
      throw new InternalError();
    }

    forEachNonZero(
        new IntIntDoubleFunction() {
          @Override
          public double apply(int i, int j, double value) {
            if (transposeA) {
              int tmp = i;
              i = j;
              j = tmp;
            }
            zElements[zi + zStride * i] += value * yElements[yi + yStride * j];
            return value;
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
      throw new IllegalArgumentException("Matrix2D inner dimensions must agree:" + toStringShort() + ", " +
          (transposeB ? B.viewDice() : B).toStringShort());
    }
    if (C.rows != m || C.columns != p) {
      throw new IllegalArgumentException(
          "Incompatibel result matrix: " + toStringShort() + ", " + (transposeB ? B.viewDice() : B).toStringShort() +
              ", " + C.toStringShort());
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

    final org.apache.mahout.math.jet.math.PlusMult fun = org.apache.mahout.math.jet.math.PlusMult.plusMult(0);

    forEachNonZero(
        new IntIntDoubleFunction() {
          @Override
          public double apply(int i, int j, double value) {
            fun.setMultiplicator(value * alpha);
            if (!transposeA) {
              Crows[i].assign(Brows[j], fun);
            } else {
              Crows[j].assign(Brows[i], fun);
            }
            return value;
          }
        }
    );

    return C;
  }
}
