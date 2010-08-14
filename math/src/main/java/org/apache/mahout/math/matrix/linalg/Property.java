/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.Sorting;
import org.apache.mahout.math.Swapper;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.IntComparator;
import org.apache.mahout.math.list.ObjectArrayList;
import org.apache.mahout.math.matrix.DoubleFactory2D;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.DoubleMatrix3D;
import org.apache.mahout.math.matrix.impl.AbstractFormatter;
import org.apache.mahout.math.matrix.impl.AbstractMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Property extends PersistentObject {

  /** The default Property object; currently has <tt>tolerance()==1.0E-9</tt>. */
  public static final Property DEFAULT = new Property(1.0E-9);

  /** A Property object with <tt>tolerance()==0.0</tt>. */
  public static final Property ZERO = new Property(0.0);

  /** A Property object with <tt>tolerance()==1.0E-12</tt>. */
  public static final Property TWELVE = new Property(1.0E-12);

  private double tolerance;

  /** Not instantiable by no-arg constructor. */
  private Property() {
    this(1.0E-9); // just to be on the safe side
  }

  /** Constructs an instance with a tolerance of <tt>Math.abs(newTolerance)</tt>. */
  public Property(double newTolerance) {
    tolerance = Math.abs(newTolerance);
  }

  /** Returns a String with <tt>length</tt> blanks. */
  protected static String blanks(int length) {
    if (length < 0) {
      length = 0;
    }
    StringBuilder buf = new StringBuilder(length);
    for (int k = 0; k < length; k++) {
      buf.append(' ');
    }
    return buf.toString();
  }

  /**
   * Checks whether the given matrix <tt>A</tt> is <i>rectangular</i>.
   *
   * @throws IllegalArgumentException if <tt>A.rows() < A.columns()</tt>.
   */
  public static void checkRectangular(AbstractMatrix2D A) {
    if (A.rows() < A.columns()) {
      throw new IllegalArgumentException("Matrix must be rectangular: " + AbstractFormatter.shape(A));
    }
  }

  /**
   * Checks whether the given matrix <tt>A</tt> is <i>square</i>.
   *
   * @throws IllegalArgumentException if <tt>A.rows() != A.columns()</tt>.
   */
  public static void checkSquare(AbstractMatrix2D A) {
    if (A.rows() != A.columns()) {
      throw new IllegalArgumentException("Matrix must be square: " + AbstractFormatter.shape(A));
    }
  }

  /** Returns the matrix's fraction of non-zero cells; <tt>A.cardinality() / A.size()</tt>. */
  public static double density(DoubleMatrix2D A) {
    return A.cardinality() / (double) A.size();
  }

  /**
   * Returns whether all cells of the given matrix <tt>A</tt> are equal to the given value. The result is <tt>true</tt>
   * if and only if <tt>A != null</tt> and <tt>! (Math.abs(value - A[i]) > tolerance())</tt> holds for all coordinates.
   *
   * @param A     the first matrix to compare.
   * @param value the value to compare against.
   * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix1D A, double value) {
    if (A == null) {
      return false;
    }
    double epsilon = tolerance();
    for (int i = A.size(); --i >= 0;) {
      //if (!(A.getQuick(i) == value)) return false;
      //if (Math.abs(value - A.getQuick(i)) > epsilon) return false;
      double x = A.getQuick(i);
      double diff = Math.abs(value - x);
      if ((diff != diff) && ((value != value && x != x) || value == x)) {
        diff = 0;
      }
      if (!(diff <= epsilon)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns whether both given matrices <tt>A</tt> and <tt>B</tt> are equal. The result is <tt>true</tt> if
   * <tt>A==B</tt>. Otherwise, the result is <tt>true</tt> if and only if both arguments are <tt>!= null</tt>, have the
   * same size and <tt>! (Math.abs(A[i] - B[i]) > tolerance())</tt> holds for all indexes.
   *
   * @param A the first matrix to compare.
   * @param B the second matrix to compare.
   * @return <tt>true</tt> if both matrices are equal; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix1D A, DoubleMatrix1D B) {
    if (A == B) {
      return true;
    }
    if (!(A != null && B != null)) {
      return false;
    }
    int size = A.size();
    if (size != B.size()) {
      return false;
    }

    double epsilon = tolerance();
    for (int i = size; --i >= 0;) {
      //if (!(getQuick(i) == B.getQuick(i))) return false;
      //if (Math.abs(A.getQuick(i) - B.getQuick(i)) > epsilon) return false;
      double x = A.getQuick(i);
      double value = B.getQuick(i);
      double diff = Math.abs(value - x);
      if ((diff != diff) && ((value != value && x != x) || value == x)) {
        diff = 0;
      }
      if (!(diff <= epsilon)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns whether all cells of the given matrix <tt>A</tt> are equal to the given value. The result is <tt>true</tt>
   * if and only if <tt>A != null</tt> and <tt>! (Math.abs(value - A[row,col]) > tolerance())</tt> holds for all
   * coordinates.
   *
   * @param A     the first matrix to compare.
   * @param value the value to compare against.
   * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix2D A, double value) {
    if (A == null) {
      return false;
    }
    int rows = A.rows();
    int columns = A.columns();

    double epsilon = tolerance();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        //if (!(A.getQuick(row,column) == value)) return false;
        //if (Math.abs(value - A.getQuick(row,column)) > epsilon) return false;
        double x = A.getQuick(row, column);
        double diff = Math.abs(value - x);
        if ((diff != diff) && ((value != value && x != x) || value == x)) {
          diff = 0;
        }
        if (!(diff <= epsilon)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Returns whether both given matrices <tt>A</tt> and <tt>B</tt> are equal. The result is <tt>true</tt> if
   * <tt>A==B</tt>. Otherwise, the result is <tt>true</tt> if and only if both arguments are <tt>!= null</tt>, have the
   * same number of columns and rows and <tt>! (Math.abs(A[row,col] - B[row,col]) > tolerance())</tt> holds for all
   * coordinates.
   *
   * @param A the first matrix to compare.
   * @param B the second matrix to compare.
   * @return <tt>true</tt> if both matrices are equal; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix2D A, DoubleMatrix2D B) {
    if (A == B) {
      return true;
    }
    if (!(A != null && B != null)) {
      return false;
    }
    int rows = A.rows();
    int columns = A.columns();
    if (columns != B.columns() || rows != B.rows()) {
      return false;
    }

    double epsilon = tolerance();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        //if (!(A.getQuick(row,column) == B.getQuick(row,column))) return false;
        //if (Math.abs((A.getQuick(row,column) - B.getQuick(row,column)) > epsilon) return false;
        double x = A.getQuick(row, column);
        double value = B.getQuick(row, column);
        double diff = Math.abs(value - x);
        if ((diff != diff) && ((value != value && x != x) || value == x)) {
          diff = 0;
        }
        if (!(diff <= epsilon)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Returns whether all cells of the given matrix <tt>A</tt> are equal to the given value. The result is <tt>true</tt>
   * if and only if <tt>A != null</tt> and <tt>! (Math.abs(value - A[slice,row,col]) > tolerance())</tt> holds for all
   * coordinates.
   *
   * @param A     the first matrix to compare.
   * @param value the value to compare against.
   * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix3D A, double value) {
    if (A == null) {
      return false;
    }
    int rows = A.rows();
    int columns = A.columns();

    double epsilon = tolerance();
    for (int slice = A.slices(); --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns; --column >= 0;) {
          //if (!(A.getQuick(slice,row,column) == value)) return false;
          //if (Math.abs(value - A.getQuick(slice,row,column)) > epsilon) return false;
          double x = A.getQuick(slice, row, column);
          double diff = Math.abs(value - x);
          if ((diff != diff) && ((value != value && x != x) || value == x)) {
            diff = 0;
          }
          if (!(diff <= epsilon)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  /**
   * Returns whether both given matrices <tt>A</tt> and <tt>B</tt> are equal. The result is <tt>true</tt> if
   * <tt>A==B</tt>. Otherwise, the result is <tt>true</tt> if and only if both arguments are <tt>!= null</tt>, have the
   * same number of columns, rows and slices, and <tt>! (Math.abs(A[slice,row,col] - B[slice,row,col]) >
   * tolerance())</tt> holds for all coordinates.
   *
   * @param A the first matrix to compare.
   * @param B the second matrix to compare.
   * @return <tt>true</tt> if both matrices are equal; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix3D A, DoubleMatrix3D B) {
    if (A == B) {
      return true;
    }
    if (!(A != null && B != null)) {
      return false;
    }
    int slices = A.slices();
    int rows = A.rows();
    int columns = A.columns();
    if (columns != B.columns() || rows != B.rows() || slices != B.slices()) {
      return false;
    }

    double epsilon = tolerance();
    for (int slice = slices; --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns; --column >= 0;) {
          //if (!(A.getQuick(slice,row,column) == B.getQuick(slice,row,column))) return false;
          //if (Math.abs(A.getQuick(slice,row,column) - B.getQuick(slice,row,column)) > epsilon) return false;
          double x = A.getQuick(slice, row, column);
          double value = B.getQuick(slice, row, column);
          double diff = Math.abs(value - x);
          if ((diff != diff) && ((value != value && x != x) || value == x)) {
            diff = 0;
          }
          if (!(diff <= epsilon)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  /**
   * Modifies the given matrix square matrix <tt>A</tt> such that it is diagonally dominant by row and column, hence
   * non-singular, hence invertible. For testing purposes only.
   *
   * @param A the square matrix to modify.
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   */
  public static void generateNonSingular(DoubleMatrix2D A) {
    checkSquare(A);
    int min = Math.min(A.rows(), A.columns());
    for (int i = min; --i >= 0;) {
      A.setQuick(i, i, 0);
    }
    for (int i = min; --i >= 0;) {
      double rowSum = A.viewRow(i).aggregate(Functions.plus, Functions.abs);
      double colSum = A.viewColumn(i).aggregate(Functions.plus, Functions.abs);
      A.setQuick(i, i, Math.max(rowSum, colSum) + i + 1);
    }
  }

  protected static String get(ObjectArrayList<String> list, int index) {
    return (list.get(index));
  }

  /**
   * A matrix <tt>A</tt> is <i>diagonal</i> if <tt>A[i,j] == 0</tt> whenever <tt>i != j</tt>. Matrix may but need not be
   * square.
   */
  public boolean isDiagonal(DoubleMatrix2D A) {
    double epsilon = tolerance();
    int rows = A.rows();
    int columns = A.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        //if (row!=column && A.getQuick(row,column) != 0) return false;
        if (row != column && !(Math.abs(A.getQuick(row, column)) <= epsilon)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>diagonally dominant by column</i> if the absolute value of each diagonal element is
   * larger than the sum of the absolute values of the off-diagonal elements in the corresponding column. <tt>returns
   * true if for all i: abs(A[i,i]) &gt; Sum(abs(A[j,i])); j != i.</tt> Matrix may but need not be square. <p> Note:
   * Ignores tolerance.
   */
  public static boolean isDiagonallyDominantByColumn(DoubleMatrix2D A) {
    //double epsilon = tolerance();
    int min = Math.min(A.rows(), A.columns());
    for (int i = min; --i >= 0;) {
      double diag = Math.abs(A.getQuick(i, i));
      diag += diag;
      if (diag <= A.viewColumn(i).aggregate(Functions.plus, Functions.abs)) {
        return false;
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>diagonally dominant by row</i> if the absolute value of each diagonal element is larger
   * than the sum of the absolute values of the off-diagonal elements in the corresponding row. <tt>returns true if for
   * all i: abs(A[i,i]) &gt; Sum(abs(A[i,j])); j != i.</tt> Matrix may but need not be square. <p> Note: Ignores
   * tolerance.
   */
  public static boolean isDiagonallyDominantByRow(DoubleMatrix2D A) {
    //double epsilon = tolerance();
    int min = Math.min(A.rows(), A.columns());
    for (int i = min; --i >= 0;) {
      double diag = Math.abs(A.getQuick(i, i));
      diag += diag;
      if (diag <= A.viewRow(i).aggregate(Functions.plus, Functions.abs)) {
        return false;
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is an <i>identity</i> matrix if <tt>A[i,i] == 1</tt> and all other cells are zero. Matrix may
   * but need not be square.
   */
  public boolean isIdentity(DoubleMatrix2D A) {
    double epsilon = tolerance();
    int rows = A.rows();
    int columns = A.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        double v = A.getQuick(row, column);
        if (row == column) {
          if (!(Math.abs(1 - v) < epsilon)) {
            return false;
          }
        } else if (!(Math.abs(v) <= epsilon)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>lower bidiagonal</i> if <tt>A[i,j]==0</tt> unless <tt>i==j || i==j+1</tt>. Matrix may but
   * need not be square.
   */
  public boolean isLowerBidiagonal(DoubleMatrix2D A) {
    double epsilon = tolerance();
    int rows = A.rows();
    int columns = A.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (!(row == column || row == column + 1)) {
          //if (A.getQuick(row,column) != 0) return false;
          if (!(Math.abs(A.getQuick(row, column)) <= epsilon)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>lower triangular</i> if <tt>A[i,j]==0</tt> whenever <tt>i &lt; j</tt>. Matrix may but
   * need not be square.
   */
  public boolean isLowerTriangular(DoubleMatrix2D A) {
    double epsilon = tolerance();
    int rows = A.rows();
    int columns = A.columns();
    for (int column = columns; --column >= 0;) {
      for (int row = Math.min(column, rows); --row >= 0;) {
        //if (A.getQuick(row,column) != 0) return false;
        if (!(Math.abs(A.getQuick(row, column)) <= epsilon)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>non-negative</i> if <tt>A[i,j] &gt;= 0</tt> holds for all cells. <p> Note: Ignores
   * tolerance.
   */
  public static boolean isNonNegative(DoubleMatrix2D A) {
    int rows = A.rows();
    int columns = A.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (!(A.getQuick(row, column) >= 0)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A square matrix <tt>A</tt> is <i>orthogonal</i> if <tt>A*transpose(A) = I</tt>.
   *
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   */
  public boolean isOrthogonal(DoubleMatrix2D A) {
    checkSquare(A);
    return equals(A.zMult(A, null, 1, 0, false, true),
        DoubleFactory2D.dense.identity(A.rows()));
  }

  /** A matrix <tt>A</tt> is <i>positive</i> if <tt>A[i,j] &gt; 0</tt> holds for all cells. <p> Note: Ignores tolerance. */
  public static boolean isPositive(DoubleMatrix2D A) {
    int rows = A.rows();
    int columns = A.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (!(A.getQuick(row, column) > 0)) {
          return false;
        }
      }
    }
    return true;
  }

  /** A matrix <tt>A</tt> is <i>singular</i> if it has no inverse, that is, iff <tt>det(A)==0</tt>. */
  public boolean isSingular(DoubleMatrix2D A) {
    return Math.abs(Algebra.det(A)) < tolerance();
  }

  /**
   * A square matrix <tt>A</tt> is <i>skew-symmetric</i> if <tt>A = -transpose(A)</tt>, that is <tt>A[i,j] ==
   * -A[j,i]</tt>.
   *
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   */
  public boolean isSkewSymmetric(DoubleMatrix2D A) {
    checkSquare(A);
    double epsilon = tolerance();
    int rows = A.rows();
    //int columns = A.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = rows; --column >= 0;) {
        //if (A.getQuick(row,column) != -A.getQuick(column,row)) return false;
        if (!(Math.abs(A.getQuick(row, column) + A.getQuick(column, row)) <= epsilon)) {
          return false;
        }
      }
    }
    return true;
  }

  /** A matrix <tt>A</tt> is <i>square</i> if it has the same number of rows and columns. */
  public static boolean isSquare(AbstractMatrix2D A) {
    return A.rows() == A.columns();
  }

  /**
   * A matrix <tt>A</tt> is <i>strictly lower triangular</i> if <tt>A[i,j]==0</tt> whenever <tt>i &lt;= j</tt>. Matrix
   * may but need not be square.
   */
  public boolean isStrictlyLowerTriangular(DoubleMatrix2D A) {
    double epsilon = tolerance();
    int rows = A.rows();
    int columns = A.columns();
    for (int column = columns; --column >= 0;) {
      for (int row = Math.min(rows, column + 1); --row >= 0;) {
        //if (A.getQuick(row,column) != 0) return false;
        if (!(Math.abs(A.getQuick(row, column)) <= epsilon)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>strictly triangular</i> if it is triangular and its diagonal elements all equal 0. Matrix
   * may but need not be square.
   */
  public boolean isStrictlyTriangular(DoubleMatrix2D A) {
    if (!isTriangular(A)) {
      return false;
    }

    double epsilon = tolerance();
    for (int i = Math.min(A.rows(), A.columns()); --i >= 0;) {
      //if (A.getQuick(i,i) != 0) return false;
      if (!(Math.abs(A.getQuick(i, i)) <= epsilon)) {
        return false;
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>strictly upper triangular</i> if <tt>A[i,j]==0</tt> whenever <tt>i &gt;= j</tt>. Matrix
   * may but need not be square.
   */
  public boolean isStrictlyUpperTriangular(DoubleMatrix2D A) {
    double epsilon = tolerance();
    int rows = A.rows();
    int columns = A.columns();
    for (int column = columns; --column >= 0;) {
      for (int row = rows; --row >= column;) {
        //if (A.getQuick(row,column) != 0) return false;
        if (!(Math.abs(A.getQuick(row, column)) <= epsilon)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>symmetric</i> if <tt>A = tranpose(A)</tt>, that is <tt>A[i,j] == A[j,i]</tt>.
   *
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   */
  public boolean isSymmetric(DoubleMatrix2D A) {
    checkSquare(A);
    return equals(A, A.viewDice());
  }

  /**
   * A matrix <tt>A</tt> is <i>triangular</i> iff it is either upper or lower triangular. Matrix may but need not be
   * square.
   */
  public boolean isTriangular(DoubleMatrix2D A) {
    return isLowerTriangular(A) || isUpperTriangular(A);
  }

  /**
   * A matrix <tt>A</tt> is <i>tridiagonal</i> if <tt>A[i,j]==0</tt> whenever <tt>Math.abs(i-j) > 1</tt>. Matrix may but
   * need not be square.
   */
  public boolean isTridiagonal(DoubleMatrix2D A) {
    double epsilon = tolerance();
    int rows = A.rows();
    int columns = A.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (Math.abs(row - column) > 1) {
          //if (A.getQuick(row,column) != 0) return false;
          if (!(Math.abs(A.getQuick(row, column)) <= epsilon)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>unit triangular</i> if it is triangular and its diagonal elements all equal 1. Matrix may
   * but need not be square.
   */
  public boolean isUnitTriangular(DoubleMatrix2D A) {
    if (!isTriangular(A)) {
      return false;
    }

    double epsilon = tolerance();
    for (int i = Math.min(A.rows(), A.columns()); --i >= 0;) {
      //if (A.getQuick(i,i) != 1) return false;
      if (!(Math.abs(1 - A.getQuick(i, i)) <= epsilon)) {
        return false;
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>upper bidiagonal</i> if <tt>A[i,j]==0</tt> unless <tt>i==j || i==j-1</tt>. Matrix may but
   * need not be square.
   */
  public boolean isUpperBidiagonal(DoubleMatrix2D A) {
    double epsilon = tolerance();
    int rows = A.rows();
    int columns = A.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (!(row == column || row == column - 1)) {
          //if (A.getQuick(row,column) != 0) return false;
          if (!(Math.abs(A.getQuick(row, column)) <= epsilon)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>upper triangular</i> if <tt>A[i,j]==0</tt> whenever <tt>i &gt; j</tt>. Matrix may but
   * need not be square.
   */
  public boolean isUpperTriangular(DoubleMatrix2D A) {
    double epsilon = tolerance();
    int rows = A.rows();
    int columns = A.columns();
    for (int column = columns; --column >= 0;) {
      for (int row = rows; --row > column;) {
        //if (A.getQuick(row,column) != 0) return false;
        if (!(Math.abs(A.getQuick(row, column)) <= epsilon)) {
          return false;
        }
      }
    }
    return true;
  }

  /** A matrix <tt>A</tt> is <i>zero</i> if all its cells are zero. */
  public boolean isZero(DoubleMatrix2D A) {
    return equals(A, 0);
  }

  /**
   * The <i>lower bandwidth</i> of a square matrix <tt>A</tt> is the maximum <tt>i-j</tt> for which <tt>A[i,j]</tt> is
   * nonzero and <tt>i &gt; j</tt>. A <i>banded</i> matrix has a "band" about the diagonal. Diagonal, tridiagonal and
   * triangular matrices are special cases.
   *
   * @param A the square matrix to analyze.
   * @return the lower bandwith.
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   * @see #semiBandwidth(DoubleMatrix2D)
   * @see #upperBandwidth(DoubleMatrix2D)
   */
  public int lowerBandwidth(DoubleMatrix2D A) {
    checkSquare(A);
    double epsilon = tolerance();
    int rows = A.rows();

    for (int k = rows; --k >= 0;) {
      for (int i = rows - k; --i >= 0;) {
        int j = i + k;
        //if (A.getQuick(j,i) != 0) return k;
        if (!(Math.abs(A.getQuick(j, i)) <= epsilon)) {
          return k;
        }
      }
    }
    return 0;
  }

  /**
   * Returns the <i>semi-bandwidth</i> of the given square matrix <tt>A</tt>. A <i>banded</i> matrix has a "band" about
   * the diagonal. It is a matrix with all cells equal to zero, with the possible exception of the cells along the
   * diagonal line, the <tt>k</tt> diagonal lines above the diagonal, and the <tt>k</tt> diagonal lines below the
   * diagonal. The <i>semi-bandwith l</i> is the number <tt>k+1</tt>. The <i>bandwidth p</i> is the number <tt>2*k +
   * 1</tt>. For example, a tridiagonal matrix corresponds to <tt>k=1, l=2, p=3</tt>, a diagonal or zero matrix
   * corresponds to <tt>k=0, l=1, p=1</tt>, <p> The <i>upper bandwidth</i> is the maximum <tt>j-i</tt> for which
   * <tt>A[i,j]</tt> is nonzero and <tt>j &gt; i</tt>. The <i>lower bandwidth</i> is the maximum <tt>i-j</tt> for which
   * <tt>A[i,j]</tt> is nonzero and <tt>i &gt; j</tt>. Diagonal, tridiagonal and triangular matrices are special cases.
   * <p> Examples: <table border="1" cellspacing="0"> <tr align="left" valign="top"> <td valign="middle"
   * align="left"><tt>matrix</tt></td> <td> <tt>4&nbsp;x&nbsp;4&nbsp;<br> 0&nbsp;0&nbsp;0&nbsp;0<br>
   * 0&nbsp;0&nbsp;0&nbsp;0<br> 0&nbsp;0&nbsp;0&nbsp;0<br> 0&nbsp;0&nbsp;0&nbsp;0 </tt></td> <td><tt>4&nbsp;x&nbsp;4<br>
   * 1&nbsp;0&nbsp;0&nbsp;0<br> 0&nbsp;0&nbsp;0&nbsp;0<br> 0&nbsp;0&nbsp;0&nbsp;0<br> 0&nbsp;0&nbsp;0&nbsp;1 </tt></td>
   * <td><tt>4&nbsp;x&nbsp;4<br> 1&nbsp;1&nbsp;0&nbsp;0<br> 1&nbsp;1&nbsp;1&nbsp;0<br> 0&nbsp;1&nbsp;1&nbsp;1<br>
   * 0&nbsp;0&nbsp;1&nbsp;1 </tt></td> <td><tt> 4&nbsp;x&nbsp;4<br> 0&nbsp;1&nbsp;1&nbsp;1<br>
   * 0&nbsp;1&nbsp;1&nbsp;1<br> 0&nbsp;0&nbsp;0&nbsp;1<br> 0&nbsp;0&nbsp;0&nbsp;1 </tt></td> <td><tt>
   * 4&nbsp;x&nbsp;4<br> 0&nbsp;0&nbsp;0&nbsp;0<br> 1&nbsp;1&nbsp;0&nbsp;0<br> 1&nbsp;1&nbsp;0&nbsp;0<br>
   * 1&nbsp;1&nbsp;1&nbsp;1 </tt></td> <td><tt>4&nbsp;x&nbsp;4<br> 1&nbsp;1&nbsp;0&nbsp;0<br> 0&nbsp;1&nbsp;1&nbsp;0<br>
   * 0&nbsp;1&nbsp;0&nbsp;1<br> 1&nbsp;0&nbsp;1&nbsp;1 </tt><tt> </tt> </td> <td><tt>4&nbsp;x&nbsp;4<br>
   * 1&nbsp;1&nbsp;1&nbsp;0<br> 0&nbsp;1&nbsp;0&nbsp;0<br> 1&nbsp;1&nbsp;0&nbsp;1<br> 0&nbsp;0&nbsp;1&nbsp;1 </tt> </td>
   * </tr> <tr align="center" valign="middle"> <td><tt>upperBandwidth</tt></td> <td> <div
   * align="center"><tt>0</tt></div> </td> <td> <div align="center"><tt>0</tt></div> </td> <td> <div
   * align="center"><tt>1</tt></div> </td> <td><tt>3</tt></td> <td align="center" valign="middle"><tt>0</tt></td> <td
   * align="center" valign="middle"> <div align="center"><tt>1</tt></div> </td> <td align="center" valign="middle"> <div
   * align="center"><tt>2</tt></div> </td> </tr> <tr align="center" valign="middle"> <td><tt>lowerBandwidth</tt></td>
   * <td> <div align="center"><tt>0</tt></div> </td> <td> <div align="center"><tt>0</tt></div> </td> <td> <div
   * align="center"><tt>1</tt></div> </td> <td><tt>0</tt></td> <td align="center" valign="middle"><tt>3</tt></td> <td
   * align="center" valign="middle"> <div align="center"><tt>3</tt></div> </td> <td align="center" valign="middle"> <div
   * align="center"><tt>2</tt></div> </td> </tr> <tr align="center" valign="middle"> <td><tt>semiBandwidth</tt></td>
   * <td> <div align="center"><tt>1</tt></div> </td> <td> <div align="center"><tt>1</tt></div> </td> <td> <div
   * align="center"><tt>2</tt></div> </td> <td><tt>4</tt></td> <td align="center" valign="middle"><tt>4</tt></td> <td
   * align="center" valign="middle"> <div align="center"><tt>4</tt></div> </td> <td align="center" valign="middle"> <div
   * align="center"><tt>3</tt></div> </td> </tr> <tr align="center" valign="middle"> <td><tt>description</tt></td> <td>
   * <div align="center"><tt>zero</tt></div> </td> <td> <div align="center"><tt>diagonal</tt></div> </td> <td> <div
   * align="center"><tt>tridiagonal</tt></div> </td> <td><tt>upper triangular</tt></td> <td align="center"
   * valign="middle"><tt>lower triangular</tt></td> <td align="center" valign="middle"> <div
   * align="center"><tt>unstructured</tt></div> </td> <td align="center" valign="middle"> <div
   * align="center"><tt>unstructured</tt></div> </td> </tr> </table>
   *
   * @param A the square matrix to analyze.
   * @return the semi-bandwith <tt>l</tt>.
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   * @see #lowerBandwidth(DoubleMatrix2D)
   * @see #upperBandwidth(DoubleMatrix2D)
   */
  public int semiBandwidth(DoubleMatrix2D A) {
    checkSquare(A);
    double epsilon = tolerance();
    int rows = A.rows();

    for (int k = rows; --k >= 0;) {
      for (int i = rows - k; --i >= 0;) {
        int j = i + k;
        //if (A.getQuick(j,i) != 0) return k+1;
        //if (A.getQuick(i,j) != 0) return k+1;
        if (!(Math.abs(A.getQuick(j, i)) <= epsilon)) {
          return k + 1;
        }
        if (!(Math.abs(A.getQuick(i, j)) <= epsilon)) {
          return k + 1;
        }
      }
    }
    return 1;
  }

  /**
   * Sets the tolerance to <tt>Math.abs(newTolerance)</tt>.
   *
   * @throws UnsupportedOperationException if <tt>this==DEFAULT || this==ZERO || this==TWELVE</tt>.
   */
  public void setTolerance(double newTolerance) {
    if (this == DEFAULT || this == ZERO || this == TWELVE) {
      throw new IllegalArgumentException("Attempted to modify immutable object.");
      //throw new UnsupportedOperationException("Attempted to modify object."); // since JDK1.2
    }
    tolerance = Math.abs(newTolerance);
  }

  /** Returns the current tolerance. */
  public double tolerance() {
    return tolerance;
  }

  /**
   * Returns summary information about the given matrix <tt>A</tt>. That is a String with (propertyName, propertyValue)
   * pairs. Useful for debugging or to quickly get the rough picture of a matrix. For example,
   * <pre>
   * density                      : 0.9
   * isDiagonal                   : false
   * isDiagonallyDominantByRow    : false
   * isDiagonallyDominantByColumn : false
   * isIdentity                   : false
   * isLowerBidiagonal            : false
   * isLowerTriangular            : false
   * isNonNegative                : true
   * isOrthogonal                 : Illegal operation or error: Matrix must be square.
   * isPositive                   : true
   * isSingular                   : Illegal operation or error: Matrix must be square.
   * isSkewSymmetric              : Illegal operation or error: Matrix must be square.
   * isSquare                     : false
   * isStrictlyLowerTriangular    : false
   * isStrictlyTriangular         : false
   * isStrictlyUpperTriangular    : false
   * isSymmetric                  : Illegal operation or error: Matrix must be square.
   * isTriangular                 : false
   * isTridiagonal                : false
   * isUnitTriangular             : false
   * isUpperBidiagonal            : false
   * isUpperTriangular            : false
   * isZero                       : false
   * lowerBandwidth               : Illegal operation or error: Matrix must be square.
   * semiBandwidth                : Illegal operation or error: Matrix must be square.
   * upperBandwidth               : Illegal operation or error: Matrix must be square.
   * </pre>
   */
  public String toString(DoubleMatrix2D A) {
    final ObjectArrayList<String> names = new ObjectArrayList<String>();
    final ObjectArrayList<String> values = new ObjectArrayList<String>();

    // determine properties
    names.add("density");
    String unknown = "Illegal operation or error: ";
    try {
      values.add(String.valueOf(density(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    // determine properties
    names.add("isDiagonal");
    try {
      values.add(String.valueOf(isDiagonal(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    // determine properties
    names.add("isDiagonallyDominantByRow");
    try {
      values.add(String.valueOf(isDiagonallyDominantByRow(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    // determine properties
    names.add("isDiagonallyDominantByColumn");
    try {
      values.add(String.valueOf(isDiagonallyDominantByColumn(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isIdentity");
    try {
      values.add(String.valueOf(isIdentity(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isLowerBidiagonal");
    try {
      values.add(String.valueOf(isLowerBidiagonal(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isLowerTriangular");
    try {
      values.add(String.valueOf(isLowerTriangular(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isNonNegative");
    try {
      values.add(String.valueOf(isNonNegative(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isOrthogonal");
    try {
      values.add(String.valueOf(isOrthogonal(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isPositive");
    try {
      values.add(String.valueOf(isPositive(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isSingular");
    try {
      values.add(String.valueOf(isSingular(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isSkewSymmetric");
    try {
      values.add(String.valueOf(isSkewSymmetric(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isSquare");
    try {
      values.add(String.valueOf(isSquare(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isStrictlyLowerTriangular");
    try {
      values.add(String.valueOf(isStrictlyLowerTriangular(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isStrictlyTriangular");
    try {
      values.add(String.valueOf(isStrictlyTriangular(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isStrictlyUpperTriangular");
    try {
      values.add(String.valueOf(isStrictlyUpperTriangular(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isSymmetric");
    try {
      values.add(String.valueOf(isSymmetric(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isTriangular");
    try {
      values.add(String.valueOf(isTriangular(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isTridiagonal");
    try {
      values.add(String.valueOf(isTridiagonal(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isUnitTriangular");
    try {
      values.add(String.valueOf(isUnitTriangular(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isUpperBidiagonal");
    try {
      values.add(String.valueOf(isUpperBidiagonal(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isUpperTriangular");
    try {
      values.add(String.valueOf(isUpperTriangular(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("isZero");
    try {
      values.add(String.valueOf(isZero(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("lowerBandwidth");
    try {
      values.add(String.valueOf(lowerBandwidth(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("semiBandwidth");
    try {
      values.add(String.valueOf(semiBandwidth(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("upperBandwidth");
    try {
      values.add(String.valueOf(upperBandwidth(A)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }


    // sort ascending by property name
    IntComparator comp = new IntComparator() {
      public int compare(int a, int b) {
        return get(names, a).compareTo(get(names, b));
      }
    };
    Swapper swapper = new Swapper() {
      public void swap(int a, int b) {
        String tmp = names.get(a);
        names.set(a, names.get(b));
        names.set(b, tmp);
        tmp = values.get(a);
        values.set(a, values.get(b));
        values.set(b, tmp);
      }
    };
    Sorting.quickSort(0, names.size(), comp, swapper);

    // determine padding for nice formatting
    int maxLength = 0;
    for (int i = 0; i < names.size(); i++) {
      int length = names.get(i).length();
      maxLength = Math.max(length, maxLength);
    }

    // finally, format properties
    StringBuilder buf = new StringBuilder();
    for (int i = 0; i < names.size(); i++) {
      String name = names.get(i);
      buf.append(name);
      buf.append(blanks(maxLength - name.length()));
      buf.append(" : ");
      buf.append(values.get(i));
      if (i < names.size() - 1) {
        buf.append('\n');
      }
    }

    return buf.toString();
  }

  /**
   * The <i>upper bandwidth</i> of a square matrix <tt>A</tt> is the maximum <tt>j-i</tt> for which <tt>A[i,j]</tt> is
   * nonzero and <tt>j &gt; i</tt>. A <i>banded</i> matrix has a "band" about the diagonal. Diagonal, tridiagonal and
   * triangular matrices are special cases.
   *
   * @param A the square matrix to analyze.
   * @return the upper bandwith.
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   * @see #semiBandwidth(DoubleMatrix2D)
   * @see #lowerBandwidth(DoubleMatrix2D)
   */
  public int upperBandwidth(DoubleMatrix2D A) {
    checkSquare(A);
    double epsilon = tolerance();
    int rows = A.rows();

    for (int k = rows; --k >= 0;) {
      for (int i = rows - k; --i >= 0;) {
        int j = i + k;
        //if (A.getQuick(i,j) != 0) return k;
        if (!(Math.abs(A.getQuick(i, j)) <= epsilon)) {
          return k;
        }
      }
    }
    return 0;
  }
}
