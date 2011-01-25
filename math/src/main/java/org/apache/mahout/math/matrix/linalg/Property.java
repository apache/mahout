/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.impl.AbstractMatrix2D;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix2D;

import java.util.Formatter;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public final class Property {

  /** The default Property object; currently has <tt>tolerance()==1.0E-9</tt>. */
  public static final Property DEFAULT = new Property(1.0E-9);

  /** A Property object with <tt>tolerance()==0.0</tt>. */
  public static final Property ZERO = new Property(0.0);

  private final double tolerance;

  /** Constructs an instance with a tolerance of <tt>Math.abs(newTolerance)</tt>. */
  public Property(double newTolerance) {
    tolerance = Math.abs(newTolerance);
  }

  /**
   * Checks whether the given matrix <tt>A</tt> is <i>rectangular</i>.
   *
   * @throws IllegalArgumentException if <tt>A.rows() < A.columns()</tt>.
   */
  public static void checkRectangular(AbstractMatrix2D a) {
    if (a.rows() < a.columns()) {
      throw new IllegalArgumentException("Matrix must be rectangular");
    }
  }

  /**
   * Checks whether the given matrix <tt>A</tt> is <i>square</i>.
   *
   * @throws IllegalArgumentException if <tt>A.rows() != A.columns()</tt>.
   */
  public static void checkSquare(AbstractMatrix2D a) {
    if (a.rows() != a.columns()) {
      throw new IllegalArgumentException("Matrix must be square");
    }
  }

  /** Returns the matrix's fraction of non-zero cells; <tt>A.cardinality() / A.size()</tt>. */
  public static double density(DoubleMatrix2D a) {
    return a.cardinality() / (double) a.size();
  }

  /**
   * Returns whether all cells of the given matrix <tt>A</tt> are equal to the given value. The result is <tt>true</tt>
   * if and only if <tt>A != null</tt> and <tt>! (Math.abs(value - A[i]) > tolerance())</tt> holds for all coordinates.
   *
   * @param a     the first matrix to compare.
   * @param value the value to compare against.
   * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix1D a, double value) {
    if (a == null) {
      return false;
    }
    double epsilon = tolerance();
    for (int i = a.size(); --i >= 0;) {
      //if (!(A.getQuick(i) == value)) return false;
      //if (Math.abs(value - A.getQuick(i)) > epsilon) return false;
      double x = a.getQuick(i);
      double diff = Math.abs(value - x);
      if (Double.isNaN(diff) && (Double.isNaN(value) && Double.isNaN(x) || value == x)) {
        diff = 0;
      }
      if (diff > epsilon) {
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
   * @param a the first matrix to compare.
   * @param b the second matrix to compare.
   * @return <tt>true</tt> if both matrices are equal; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix1D a, DoubleMatrix1D b) {
    if (a == b) {
      return true;
    }
    if (!(a != null && b != null)) {
      return false;
    }
    int size = a.size();
    if (size != b.size()) {
      return false;
    }

    double epsilon = tolerance();
    for (int i = size; --i >= 0;) {
      //if (!(getQuick(i) == B.getQuick(i))) return false;
      //if (Math.abs(A.getQuick(i) - B.getQuick(i)) > epsilon) return false;
      double x = a.getQuick(i);
      double value = b.getQuick(i);
      double diff = Math.abs(value - x);
      if (Double.isNaN(diff) && ((Double.isNaN(value) && Double.isNaN(x)) || value == x)) {
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
   * @param a     the first matrix to compare.
   * @param value the value to compare against.
   * @return <tt>true</tt> if the matrix is equal to the value; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix2D a, double value) {
    if (a == null) {
      return false;
    }
    int rows = a.rows();
    int columns = a.columns();

    double epsilon = tolerance();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        //if (!(A.getQuick(row,column) == value)) return false;
        //if (Math.abs(value - A.getQuick(row,column)) > epsilon) return false;
        double x = a.getQuick(row, column);
        double diff = Math.abs(value - x);
        if (Double.isNaN(diff) && (Double.isNaN(value) && Double.isNaN(x) || value == x)) {
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
   * @param a the first matrix to compare.
   * @param b the second matrix to compare.
   * @return <tt>true</tt> if both matrices are equal; <tt>false</tt> otherwise.
   */
  public boolean equals(DoubleMatrix2D a, DoubleMatrix2D b) {
    if (a == b) {
      return true;
    }
    if (!(a != null && b != null)) {
      return false;
    }
    int rows = a.rows();
    int columns = a.columns();
    if (columns != b.columns() || rows != b.rows()) {
      return false;
    }

    double epsilon = tolerance();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        //if (!(A.getQuick(row,column) == B.getQuick(row,column))) return false;
        //if (Math.abs((A.getQuick(row,column) - B.getQuick(row,column)) > epsilon) return false;
        double x = a.getQuick(row, column);
        double value = b.getQuick(row, column);
        double diff = Math.abs(value - x);
        if (Double.isNaN(diff) && ((Double.isNaN(value) && Double.isNaN(x)) || value == x)) {
          diff = 0;
        }
        if (diff > epsilon) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>diagonal</i> if <tt>A[i,j] == 0</tt> whenever <tt>i != j</tt>. Matrix may but need not be
   * square.
   */
  public boolean isDiagonal(DoubleMatrix2D a) {
    double epsilon = tolerance();
    int rows = a.rows();
    int columns = a.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        //if (row!=column && A.getQuick(row,column) != 0) return false;
        if (row != column && Math.abs(a.getQuick(row, column)) > epsilon) {
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
  public static boolean isDiagonallyDominantByColumn(DoubleMatrix2D a) {
    //double epsilon = tolerance();
    int min = Math.min(a.rows(), a.columns());
    for (int i = min; --i >= 0;) {
      double diag = Math.abs(a.getQuick(i, i));
      diag += diag;
      if (diag <= a.viewColumn(i).aggregate(Functions.PLUS, Functions.ABS)) {
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
  public static boolean isDiagonallyDominantByRow(DoubleMatrix2D a) {
    //double epsilon = tolerance();
    int min = Math.min(a.rows(), a.columns());
    for (int i = min; --i >= 0;) {
      double diag = Math.abs(a.getQuick(i, i));
      diag += diag;
      if (diag <= a.viewRow(i).aggregate(Functions.PLUS, Functions.ABS)) {
        return false;
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is an <i>identity</i> matrix if <tt>A[i,i] == 1</tt> and all other cells are zero. Matrix may
   * but need not be square.
   */
  public boolean isIdentity(DoubleMatrix2D a) {
    double epsilon = tolerance();
    int rows = a.rows();
    int columns = a.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        double v = a.getQuick(row, column);
        if (row == column) {
          if (Math.abs(1 - v) > epsilon) {
            return false;
          }
        } else if (Math.abs(v) > epsilon) {
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
  public boolean isLowerBidiagonal(DoubleMatrix2D a) {
    double epsilon = tolerance();
    int rows = a.rows();
    int columns = a.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (!(row == column || row == column + 1) && Math.abs(a.getQuick(row, column)) > epsilon) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>lower triangular</i> if <tt>A[i,j]==0</tt> whenever <tt>i &lt; j</tt>. Matrix may but
   * need not be square.
   */
  public boolean isLowerTriangular(DoubleMatrix2D a) {
    double epsilon = tolerance();
    int rows = a.rows();
    int columns = a.columns();
    for (int column = columns; --column >= 0;) {
      for (int row = Math.min(column, rows); --row >= 0;) {
        //if (A.getQuick(row,column) != 0) return false;
        if (Math.abs(a.getQuick(row, column)) > epsilon) {
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
  public static boolean isNonNegative(DoubleMatrix2D a) {
    int rows = a.rows();
    int columns = a.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (a.getQuick(row, column) < 0) {
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
  public boolean isOrthogonal(DoubleMatrix2D a) {
    checkSquare(a);
    return equals(a.zMult(a, null, 1, 0, false, true),
                  DenseDoubleMatrix2D.identity(a.rows()));
  }

  /** A matrix <tt>A</tt> is <i>positive</i> if <tt>A[i,j] &gt; 0</tt> holds for all cells.
   * <p> Note: Ignores tolerance.
   */
  public static boolean isPositive(DoubleMatrix2D a) {
    int rows = a.rows();
    int columns = a.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (a.getQuick(row, column) <= 0) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A square matrix <tt>A</tt> is <i>skew-symmetric</i> if <tt>A = -transpose(A)</tt>, that is <tt>A[i,j] ==
   * -A[j,i]</tt>.
   *
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   */
  public boolean isSkewSymmetric(DoubleMatrix2D a) {
    checkSquare(a);
    double epsilon = tolerance();
    int rows = a.rows();
    //int columns = A.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = rows; --column >= 0;) {
        //if (A.getQuick(row,column) != -A.getQuick(column,row)) return false;
        if (Math.abs(a.getQuick(row, column) + a.getQuick(column, row)) > epsilon) {
          return false;
        }
      }
    }
    return true;
  }

  /** A matrix <tt>A</tt> is <i>square</i> if it has the same number of rows and columns. */
  public static boolean isSquare(AbstractMatrix2D a) {
    return a.rows() == a.columns();
  }

  /**
   * A matrix <tt>A</tt> is <i>strictly lower triangular</i> if <tt>A[i,j]==0</tt> whenever <tt>i &lt;= j</tt>. Matrix
   * may but need not be square.
   */
  public boolean isStrictlyLowerTriangular(DoubleMatrix2D a) {
    double epsilon = tolerance();
    int rows = a.rows();
    int columns = a.columns();
    for (int column = columns; --column >= 0;) {
      for (int row = Math.min(rows, column + 1); --row >= 0;) {
        //if (A.getQuick(row,column) != 0) return false;
        if (Math.abs(a.getQuick(row, column)) > epsilon) {
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
  public boolean isStrictlyTriangular(DoubleMatrix2D a) {
    if (isTriangular(a)) {
      double epsilon = tolerance();
      for (int i = Math.min(a.rows(), a.columns()); --i >= 0;) {
        //if (A.getQuick(i,i) != 0) return false;
        if (Math.abs(a.getQuick(i, i)) > epsilon) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  /**
   * A matrix <tt>A</tt> is <i>strictly upper triangular</i> if <tt>A[i,j]==0</tt> whenever <tt>i &gt;= j</tt>. Matrix
   * may but need not be square.
   */
  public boolean isStrictlyUpperTriangular(DoubleMatrix2D a) {
    double epsilon = tolerance();
    int rows = a.rows();
    int columns = a.columns();
    for (int column = columns; --column >= 0;) {
      for (int row = rows; --row >= column;) {
        //if (A.getQuick(row,column) != 0) return false;
        if (Math.abs(a.getQuick(row, column)) > epsilon) {
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
  public boolean isSymmetric(DoubleMatrix2D a) {
    checkSquare(a);
    return equals(a, a.viewDice());
  }

  /**
   * A matrix <tt>A</tt> is <i>triangular</i> iff it is either upper or lower triangular. Matrix may but need not be
   * square.
   */
  public boolean isTriangular(DoubleMatrix2D a) {
    return isLowerTriangular(a) || isUpperTriangular(a);
  }

  /**
   * A matrix <tt>A</tt> is <i>tridiagonal</i> if <tt>A[i,j]==0</tt> whenever <tt>Math.abs(i-j) > 1</tt>. Matrix may but
   * need not be square.
   */
  public boolean isTridiagonal(DoubleMatrix2D a) {
    double epsilon = tolerance();
    int rows = a.rows();
    int columns = a.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (Math.abs(row - column) > 1 && Math.abs(a.getQuick(row, column)) > epsilon) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>unit triangular</i> if it is triangular and its diagonal elements all equal 1. Matrix may
   * but need not be square.
   */
  public boolean isUnitTriangular(DoubleMatrix2D a) {
    if (isTriangular(a)) {
      double epsilon = tolerance();
      for (int i = Math.min(a.rows(), a.columns()); --i >= 0;) {
        //if (A.getQuick(i,i) != 1) return false;
        if (Math.abs(1 - a.getQuick(i, i)) > epsilon) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  /**
   * A matrix <tt>A</tt> is <i>upper bidiagonal</i> if <tt>A[i,j]==0</tt> unless <tt>i==j || i==j-1</tt>. Matrix may but
   * need not be square.
   */
  public boolean isUpperBidiagonal(DoubleMatrix2D a) {
    double epsilon = tolerance();
    int rows = a.rows();
    int columns = a.columns();
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (!(row == column || row == column - 1) && Math.abs(a.getQuick(row, column)) > epsilon) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * A matrix <tt>A</tt> is <i>upper triangular</i> if <tt>A[i,j]==0</tt> whenever <tt>i &gt; j</tt>. Matrix may but
   * need not be square.
   */
  public boolean isUpperTriangular(DoubleMatrix2D a) {
    double epsilon = tolerance();
    int rows = a.rows();
    int columns = a.columns();
    for (int column = columns; --column >= 0;) {
      for (int row = rows; --row > column;) {
        //if (A.getQuick(row,column) != 0) return false;
        if (Math.abs(a.getQuick(row, column)) > epsilon) {
          return false;
        }
      }
    }
    return true;
  }

  /** A matrix <tt>A</tt> is <i>zero</i> if all its cells are zero. */
  public boolean isZero(DoubleMatrix2D a) {
    return equals(a, 0);
  }

  /**
   * The <i>lower bandwidth</i> of a square matrix <tt>A</tt> is the maximum <tt>i-j</tt> for which <tt>A[i,j]</tt> is
   * nonzero and <tt>i &gt; j</tt>. A <i>banded</i> matrix has a "band" about the diagonal. Diagonal, tridiagonal and
   * triangular matrices are special cases.
   *
   * @param a the square matrix to analyze.
   * @return the lower bandwith.
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   * @see #semiBandwidth(DoubleMatrix2D)
   * @see #upperBandwidth(DoubleMatrix2D)
   */
  public int lowerBandwidth(DoubleMatrix2D a) {
    checkSquare(a);
    double epsilon = tolerance();
    int rows = a.rows();

    for (int k = rows; --k >= 0;) {
      for (int i = rows - k; --i >= 0;) {
        int j = i + k;
        //if (A.getQuick(j,i) != 0) return k;
        if (Math.abs(a.getQuick(j, i)) > epsilon) {
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
   * @param a the square matrix to analyze.
   * @return the semi-bandwith <tt>l</tt>.
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   * @see #lowerBandwidth(DoubleMatrix2D)
   * @see #upperBandwidth(DoubleMatrix2D)
   */
  public int semiBandwidth(DoubleMatrix2D a) {
    checkSquare(a);
    double epsilon = tolerance();
    int rows = a.rows();

    for (int k = rows; --k >= 0;) {
      for (int i = rows - k; --i >= 0;) {
        int j = i + k;
        //if (A.getQuick(j,i) != 0) return k+1;
        //if (A.getQuick(i,j) != 0) return k+1;
        if (!(Math.abs(a.getQuick(j, i)) <= epsilon)) {
          return k + 1;
        }
        if (Math.abs(a.getQuick(i, j)) > epsilon) {
          return k + 1;
        }
      }
    }
    return 1;
  }

  /** Returns the current tolerance. */
  public double tolerance() {
    return tolerance;
  }

  /**
   * The <i>upper bandwidth</i> of a square matrix <tt>A</tt> is the maximum <tt>j-i</tt> for which <tt>A[i,j]</tt> is
   * nonzero and <tt>j &gt; i</tt>. A <i>banded</i> matrix has a "band" about the diagonal. Diagonal, tridiagonal and
   * triangular matrices are special cases.
   *
   * @param a the square matrix to analyze.
   * @return the upper bandwith.
   * @throws IllegalArgumentException if <tt>!isSquare(A)</tt>.
   * @see #semiBandwidth(DoubleMatrix2D)
   * @see #lowerBandwidth(DoubleMatrix2D)
   */
  public int upperBandwidth(DoubleMatrix2D a) {
    checkSquare(a);
    double epsilon = tolerance();
    int rows = a.rows();

    for (int k = rows; --k >= 0;) {
      for (int i = rows - k; --i >= 0;) {
        int j = i + k;
        //if (A.getQuick(i,j) != 0) return k;
        if (!(Math.abs(a.getQuick(i, j)) <= epsilon)) {
          return k;
        }
      }
    }
    return 0;
  }
}
