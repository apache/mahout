/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix;

import org.apache.mahout.matrix.PersistentObject;
import org.apache.mahout.matrix.matrix.impl.DenseObjectMatrix2D;
import org.apache.mahout.matrix.matrix.impl.SparseObjectMatrix2D;
/**
 Factory for convenient construction of 2-d matrices holding <tt>Object</tt>
 cells. Also provides convenient methods to compose (concatenate) and decompose
 (split) matrices from/to constituent blocks. </p>
 <p>&nbsp; </p>
 <table border="0" cellspacing="0">
 <tr align="left" valign="top">
 <td><i>Construction</i></td>
 <td>Use idioms like <tt>ObjectFactory2D.dense.make(4,4)</tt> to construct
 dense matrices, <tt>ObjectFactory2D.sparse.make(4,4)</tt> to construct sparse
 matrices.</td>
 </tr>
 <tr align="left" valign="top">
 <td><i> Construction with initial values </i></td>
 <td>Use other <tt>make</tt> methods to construct matrices with given initial
 values. </td>
 </tr>
 <tr align="left" valign="top">
 <td><i> Appending rows and columns </i></td>
 <td>Use methods {@link #appendColumns(ObjectMatrix2D,ObjectMatrix2D) appendColumns},
 {@link #appendColumns(ObjectMatrix2D,ObjectMatrix2D) appendRows} and {@link
#repeat(ObjectMatrix2D,int,int) repeat} to append rows and columns. </td>
 </tr>
 <tr align="left" valign="top">
 <td><i> General block matrices </i></td>
 <td>Use methods {@link #compose(ObjectMatrix2D[][]) compose} and {@link #decompose(ObjectMatrix2D[][],ObjectMatrix2D)
decompose} to work with general block matrices. </td>
 </tr>
 <tr align="left" valign="top">
 <td><i> Diagonal block matrices </i></td>
 <td>Use method {@link #composeDiagonal(ObjectMatrix2D,ObjectMatrix2D,ObjectMatrix2D)
composeDiagonal} to work with diagonal block matrices. </td>
 </tr>
 </table>
 <p>&nbsp;</p>
 <p>If the factory is used frequently it might be useful to streamline the notation.
 For example by aliasing: </p>
 <table>
 <td class="PRE">
 <pre>
 ObjectFactory2D F = ObjectFactory2D.dense;
 F.make(4,4);
 ...
 </pre>
 </td>
 </table>

 @author wolfgang.hoschek@cern.ch
 @version 1.0, 09/24/99
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class ObjectFactory2D extends PersistentObject {

  /** A factory producing dense matrices. */
  public static final ObjectFactory2D dense = new ObjectFactory2D();

  /** A factory producing sparse matrices. */
  public static final ObjectFactory2D sparse = new ObjectFactory2D();

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected ObjectFactory2D() {
  }

  /**
   * C = A||B; Constructs a new matrix which is the column-wise concatenation of two other matrices.
   * <pre>
   * 0 1 2
   * 3 4 5
   * appendColumns
   * 6 7
   * 8 9
   * -->
   * 0 1 2 6 7
   * 3 4 5 8 9
   * </pre>
   */
  public ObjectMatrix2D appendColumns(ObjectMatrix2D A, ObjectMatrix2D B) {
    // force both to have maximal shared number of rows.
    if (B.rows() > A.rows()) {
      B = B.viewPart(0, 0, A.rows(), B.columns());
    } else if (B.rows() < A.rows()) {
      A = A.viewPart(0, 0, B.rows(), A.columns());
    }

    // concatenate
    int ac = A.columns();
    int bc = B.columns();
    int r = A.rows();
    ObjectMatrix2D matrix = make(r, ac + bc);
    matrix.viewPart(0, 0, r, ac).assign(A);
    matrix.viewPart(0, ac, r, bc).assign(B);
    return matrix;
  }

  /**
   * C = A||B; Constructs a new matrix which is the row-wise concatenation of two other matrices.
   * <pre>
   * 0 1
   * 2 3
   * 4 5
   * appendRows
   * 6 7
   * 8 9
   * -->
   * 0 1
   * 2 3
   * 4 5
   * 6 7
   * 8 9
   * </pre>
   */
  public ObjectMatrix2D appendRows(ObjectMatrix2D A, ObjectMatrix2D B) {
    // force both to have maximal shared number of columns.
    if (B.columns() > A.columns()) {
      B = B.viewPart(0, 0, B.rows(), A.columns());
    } else if (B.columns() < A.columns()) {
      A = A.viewPart(0, 0, A.rows(), B.columns());
    }

    // concatenate
    int ar = A.rows();
    int br = B.rows();
    int c = A.columns();
    ObjectMatrix2D matrix = make(ar + br, c);
    matrix.viewPart(0, 0, ar, c).assign(A);
    matrix.viewPart(ar, 0, br, c).assign(B);
    return matrix;
  }

  /**
   * Checks whether the given array is rectangular, that is, whether all rows have the same number of columns.
   *
   * @throws IllegalArgumentException if the array is not rectangular.
   */
  protected static void checkRectangularShape(ObjectMatrix2D[][] array) {
    int columns = -1;
    for (int row = array.length; --row >= 0;) {
      if (array[row] != null) {
        if (columns == -1) {
          columns = array[row].length;
        }
        if (array[row].length != columns) {
          throw new IllegalArgumentException("All rows of array must have same number of columns.");
        }
      }
    }
  }

  /**
   * Checks whether the given array is rectangular, that is, whether all rows have the same number of columns.
   *
   * @throws IllegalArgumentException if the array is not rectangular.
   */
  protected static void checkRectangularShape(Object[][] array) {
    int columns = -1;
    for (int row = array.length; --row >= 0;) {
      if (array[row] != null) {
        if (columns == -1) {
          columns = array[row].length;
        }
        if (array[row].length != columns) {
          throw new IllegalArgumentException("All rows of array must have same number of columns.");
        }
      }
    }
  }

  /**
   * Constructs a block matrix made from the given parts. The inverse to method {@link #decompose(ObjectMatrix2D[][],
   * ObjectMatrix2D)}. <p> All matrices of a given column within <tt>parts</tt> must have the same number of columns.
   * All matrices of a given row within <tt>parts</tt> must have the same number of rows. Otherwise an
   * <tt>IllegalArgumentException</tt> is thrown. Note that <tt>null</tt>s within <tt>parts[row,col]</tt> are an
   * exception to this rule: they are ignored. Cells are copied. Example: <table border="1" cellspacing="0"> <tr
   * align="left" valign="top"> <td><tt>Code</tt></td> <td><tt>Result</tt></td> </tr> <tr align="left" valign="top">
   * <td>
   * <pre>
   * ObjectMatrix2D[][] parts1 =
   * {
   * &nbsp;&nbsp;&nbsp;{ null,        make(2,2,1), null        },
   * &nbsp;&nbsp;&nbsp;{ make(4,4,2), null,        make(4,3,3) },
   * &nbsp;&nbsp;&nbsp;{ null,        make(2,2,4), null        }
   * };
   * log.info(compose(parts1));
   * </pre>
   * </td> <td><tt>8&nbsp;x&nbsp;9&nbsp;matrix<br> 0&nbsp;0&nbsp;0&nbsp;0&nbsp;1&nbsp;1&nbsp;0&nbsp;0&nbsp;0<br>
   * 0&nbsp;0&nbsp;0&nbsp;0&nbsp;1&nbsp;1&nbsp;0&nbsp;0&nbsp;0<br> 2&nbsp;2&nbsp;2&nbsp;2&nbsp;0&nbsp;0&nbsp;3&nbsp;3&nbsp;3<br>
   * 2&nbsp;2&nbsp;2&nbsp;2&nbsp;0&nbsp;0&nbsp;3&nbsp;3&nbsp;3<br> 2&nbsp;2&nbsp;2&nbsp;2&nbsp;0&nbsp;0&nbsp;3&nbsp;3&nbsp;3<br>
   * 2&nbsp;2&nbsp;2&nbsp;2&nbsp;0&nbsp;0&nbsp;3&nbsp;3&nbsp;3<br> 0&nbsp;0&nbsp;0&nbsp;0&nbsp;4&nbsp;4&nbsp;0&nbsp;0&nbsp;0<br>
   * 0&nbsp;0&nbsp;0&nbsp;0&nbsp;4&nbsp;4&nbsp;0&nbsp;0&nbsp;0</tt></td> </tr> <tr align="left" valign="top"> <td>
   * <pre>
   * ObjectMatrix2D[][] parts3 =
   * {
   * &nbsp;&nbsp;&nbsp;{ identity(3),               null,                        },
   * &nbsp;&nbsp;&nbsp;{ null,                      identity(3).viewColumnFlip() },
   * &nbsp;&nbsp;&nbsp;{ identity(3).viewRowFlip(), null                         }
   * };
   * log.info("\n"+make(parts3));
   * </pre>
   * </td> <td><tt>9&nbsp;x&nbsp;6&nbsp;matrix<br> 1&nbsp;0&nbsp;0&nbsp;0&nbsp;0&nbsp;0<br>
   * 0&nbsp;1&nbsp;0&nbsp;0&nbsp;0&nbsp;0<br> 0&nbsp;0&nbsp;1&nbsp;0&nbsp;0&nbsp;0<br>
   * 0&nbsp;0&nbsp;0&nbsp;0&nbsp;0&nbsp;1<br> 0&nbsp;0&nbsp;0&nbsp;0&nbsp;1&nbsp;0<br>
   * 0&nbsp;0&nbsp;0&nbsp;1&nbsp;0&nbsp;0<br> 0&nbsp;0&nbsp;1&nbsp;0&nbsp;0&nbsp;0<br>
   * 0&nbsp;1&nbsp;0&nbsp;0&nbsp;0&nbsp;0<br> 1&nbsp;0&nbsp;0&nbsp;0&nbsp;0&nbsp;0 </tt></td> </tr> <tr align="left"
   * valign="top"> <td>
   * <pre>
   * ObjectMatrix2D A = ascending(2,2);
   * ObjectMatrix2D B = descending(2,2);
   * ObjectMatrix2D _ = null;
   *
   * ObjectMatrix2D[][] parts4 =
   * {
   * &nbsp;&nbsp;&nbsp;{ A, _, A, _ },
   * &nbsp;&nbsp;&nbsp;{ _, A, _, B }
   * };
   * log.info("\n"+make(parts4));
   * </pre>
   * </td> <td><tt>4&nbsp;x&nbsp;8&nbsp;matrix<br> 1&nbsp;2&nbsp;0&nbsp;0&nbsp;1&nbsp;2&nbsp;0&nbsp;0<br>
   * 3&nbsp;4&nbsp;0&nbsp;0&nbsp;3&nbsp;4&nbsp;0&nbsp;0<br> 0&nbsp;0&nbsp;1&nbsp;2&nbsp;0&nbsp;0&nbsp;3&nbsp;2<br>
   * 0&nbsp;0&nbsp;3&nbsp;4&nbsp;0&nbsp;0&nbsp;1&nbsp;0 </tt></td> </tr> <tr align="left" valign="top"> <td>
   * <pre>
   * ObjectMatrix2D[][] parts2 =
   * {
   * &nbsp;&nbsp;&nbsp;{ null,        make(2,2,1), null        },
   * &nbsp;&nbsp;&nbsp;{ make(4,4,2), null,        make(4,3,3) },
   * &nbsp;&nbsp;&nbsp;{ null,        make(2,3,4), null        }
   * };
   * log.info("\n"+Factory2D.make(parts2));
   * </pre>
   * </td> <td><tt>IllegalArgumentException<br> A[0,1].cols != A[2,1].cols<br> (2 != 3)</tt></td> </tr> </table>
   *
   * @throws IllegalArgumentException subject to the conditions outlined above.
   */
  public ObjectMatrix2D compose(ObjectMatrix2D[][] parts) {
    checkRectangularShape(parts);
    int rows = parts.length;
    int columns = 0;
    if (parts.length > 0) {
      columns = parts[0].length;
    }
    ObjectMatrix2D empty = make(0, 0);

    if (rows == 0 || columns == 0) {
      return empty;
    }

    // determine maximum column width of each column
    int[] maxWidths = new int[columns];
    for (int column = columns; --column >= 0;) {
      int maxWidth = 0;
      for (int row = rows; --row >= 0;) {
        ObjectMatrix2D part = parts[row][column];
        if (part != null) {
          int width = part.columns();
          if (maxWidth > 0 && width > 0 && width != maxWidth) {
            throw new IllegalArgumentException("Different number of columns.");
          }
          maxWidth = Math.max(maxWidth, width);
        }
      }
      maxWidths[column] = maxWidth;
    }

    // determine row height of each row
    int[] maxHeights = new int[rows];
    for (int row = rows; --row >= 0;) {
      int maxHeight = 0;
      for (int column = columns; --column >= 0;) {
        ObjectMatrix2D part = parts[row][column];
        if (part != null) {
          int height = part.rows();
          if (maxHeight > 0 && height > 0 && height != maxHeight) {
            throw new IllegalArgumentException("Different number of rows.");
          }
          maxHeight = Math.max(maxHeight, height);
        }
      }
      maxHeights[row] = maxHeight;
    }


    // shape of result
    int resultRows = 0;
    for (int row = rows; --row >= 0;) {
      resultRows += maxHeights[row];
    }
    int resultCols = 0;
    for (int column = columns; --column >= 0;) {
      resultCols += maxWidths[column];
    }

    ObjectMatrix2D matrix = make(resultRows, resultCols);

    // copy
    int r = 0;
    for (int row = 0; row < rows; row++) {
      int c = 0;
      for (int column = 0; column < columns; column++) {
        ObjectMatrix2D part = parts[row][column];
        if (part != null) {
          matrix.viewPart(r, c, part.rows(), part.columns()).assign(part);
        }
        c += maxWidths[column];
      }
      r += maxHeights[row];
    }

    return matrix;
  }

  /**
   * Constructs a diagonal block matrix from the given parts (the <i>direct sum</i> of two matrices). That is the
   * concatenation
   * <pre>
   * A 0
   * 0 B
   * </pre>
   * (The direct sum has <tt>A.rows()+B.rows()</tt> rows and <tt>A.columns()+B.columns()</tt> columns). Cells are
   * copied.
   *
   * @return a new matrix which is the direct sum.
   */
  public ObjectMatrix2D composeDiagonal(ObjectMatrix2D A, ObjectMatrix2D B) {
    int ar = A.rows();
    int ac = A.columns();
    int br = B.rows();
    int bc = B.columns();
    ObjectMatrix2D sum = make(ar + br, ac + bc);
    sum.viewPart(0, 0, ar, ac).assign(A);
    sum.viewPart(ar, ac, br, bc).assign(B);
    return sum;
  }

  /**
   * Constructs a diagonal block matrix from the given parts. The concatenation has the form
   * <pre>
   * A 0 0
   * 0 B 0
   * 0 0 C
   * </pre>
   * from the given parts. Cells are copied.
   */
  public ObjectMatrix2D composeDiagonal(ObjectMatrix2D A, ObjectMatrix2D B, ObjectMatrix2D C) {
    ObjectMatrix2D diag = make(A.rows() + B.rows() + C.rows(), A.columns() + B.columns() + C.columns());
    diag.viewPart(0, 0, A.rows(), A.columns()).assign(A);
    diag.viewPart(A.rows(), A.columns(), B.rows(), B.columns()).assign(B);
    diag.viewPart(A.rows() + B.rows(), A.columns() + B.columns(), C.rows(), C.columns()).assign(C);
    return diag;
  }

  /**
   * Splits a block matrix into its constituent blocks; Copies blocks of a matrix into the given parts. The inverse to
   * method {@link #compose(ObjectMatrix2D[][])}. <p> All matrices of a given column within <tt>parts</tt> must have the
   * same number of columns. All matrices of a given row within <tt>parts</tt> must have the same number of rows.
   * Otherwise an <tt>IllegalArgumentException</tt> is thrown. Note that <tt>null</tt>s within <tt>parts[row,col]</tt>
   * are an exception to this rule: they are ignored. Cells are copied. Example: <table border="1" cellspacing="0"> <tr
   * align="left" valign="top"> <td><tt>Code</tt></td> <td><tt>matrix</tt></td> <td><tt>--&gt; parts </tt></td> </tr>
   * <tr align="left" valign="top"> <td>
   * <pre>
   * ObjectMatrix2D matrix = ... ;
   * ObjectMatrix2D _ = null;
   * ObjectMatrix2D A,B,C,D;
   * A = make(2,2); B = make (4,4);
   * C = make(4,3); D = make (2,2);
   * ObjectMatrix2D[][] parts =
   * {
   * &nbsp;&nbsp;&nbsp;{ _, A, _ },
   * &nbsp;&nbsp;&nbsp;{ B, _, C },
   * &nbsp;&nbsp;&nbsp;{ _, D, _ }
   * };
   * decompose(parts,matrix);
   * log.info(&quot;\nA = &quot;+A);
   * log.info(&quot;\nB = &quot;+B);
   * log.info(&quot;\nC = &quot;+C);
   * log.info(&quot;\nD = &quot;+D);
   * </pre>
   * </td> <td><tt>8&nbsp;x&nbsp;9&nbsp;matrix<br> 9&nbsp;9&nbsp;9&nbsp;9&nbsp;1&nbsp;1&nbsp;9&nbsp;9&nbsp;9<br>
   * 9&nbsp;9&nbsp;9&nbsp;9&nbsp;1&nbsp;1&nbsp;9&nbsp;9&nbsp;9<br> 2&nbsp;2&nbsp;2&nbsp;2&nbsp;9&nbsp;9&nbsp;3&nbsp;3&nbsp;3<br>
   * 2&nbsp;2&nbsp;2&nbsp;2&nbsp;9&nbsp;9&nbsp;3&nbsp;3&nbsp;3<br> 2&nbsp;2&nbsp;2&nbsp;2&nbsp;9&nbsp;9&nbsp;3&nbsp;3&nbsp;3<br>
   * 2&nbsp;2&nbsp;2&nbsp;2&nbsp;9&nbsp;9&nbsp;3&nbsp;3&nbsp;3<br> 9&nbsp;9&nbsp;9&nbsp;9&nbsp;4&nbsp;4&nbsp;9&nbsp;9&nbsp;9<br>
   * 9&nbsp;9&nbsp;9&nbsp;9&nbsp;4&nbsp;4&nbsp;9&nbsp;9&nbsp;9</tt></td> <td> <p><tt>A = 2&nbsp;x&nbsp;2&nbsp;matrix<br>
   * 1&nbsp;1<br> 1&nbsp;1</tt></p> <p><tt>B = 4&nbsp;x&nbsp;4&nbsp;matrix<br> 2&nbsp;2&nbsp;2&nbsp;2<br>
   * 2&nbsp;2&nbsp;2&nbsp;2<br> 2&nbsp;2&nbsp;2&nbsp;2<br> 2&nbsp;2&nbsp;2&nbsp;2</tt></p> <p><tt>C =
   * 4&nbsp;x&nbsp;3&nbsp;matrix<br> 3&nbsp;3&nbsp;3<br> 3&nbsp;3&nbsp;3<br> </tt><tt>3&nbsp;3&nbsp;3<br>
   * </tt><tt>3&nbsp;3&nbsp;3</tt></p> <p><tt>D = 2&nbsp;x&nbsp;2&nbsp;matrix<br> 4&nbsp;4<br> 4&nbsp;4</tt></p> </td>
   * </tr> </table>
   *
   * @throws IllegalArgumentException subject to the conditions outlined above.
   */
  public void decompose(ObjectMatrix2D[][] parts, ObjectMatrix2D matrix) {
    checkRectangularShape(parts);
    int rows = parts.length;
    int columns = 0;
    if (parts.length > 0) {
      columns = parts[0].length;
    }
    if (rows == 0 || columns == 0) {
      return;
    }

    // determine maximum column width of each column
    int[] maxWidths = new int[columns];
    for (int column = columns; --column >= 0;) {
      int maxWidth = 0;
      for (int row = rows; --row >= 0;) {
        ObjectMatrix2D part = parts[row][column];
        if (part != null) {
          int width = part.columns();
          if (maxWidth > 0 && width > 0 && width != maxWidth) {
            throw new IllegalArgumentException("Different number of columns.");
          }
          maxWidth = Math.max(maxWidth, width);
        }
      }
      maxWidths[column] = maxWidth;
    }

    // determine row height of each row
    int[] maxHeights = new int[rows];
    for (int row = rows; --row >= 0;) {
      int maxHeight = 0;
      for (int column = columns; --column >= 0;) {
        ObjectMatrix2D part = parts[row][column];
        if (part != null) {
          int height = part.rows();
          if (maxHeight > 0 && height > 0 && height != maxHeight) {
            throw new IllegalArgumentException("Different number of rows.");
          }
          maxHeight = Math.max(maxHeight, height);
        }
      }
      maxHeights[row] = maxHeight;
    }


    // shape of result parts
    int resultRows = 0;
    for (int row = rows; --row >= 0;) {
      resultRows += maxHeights[row];
    }
    int resultCols = 0;
    for (int column = columns; --column >= 0;) {
      resultCols += maxWidths[column];
    }

    if (matrix.rows() < resultRows || matrix.columns() < resultCols) {
      throw new IllegalArgumentException("Parts larger than matrix.");
    }

    // copy
    int r = 0;
    for (int row = 0; row < rows; row++) {
      int c = 0;
      for (int column = 0; column < columns; column++) {
        ObjectMatrix2D part = parts[row][column];
        if (part != null) {
          part.assign(matrix.viewPart(r, c, part.rows(), part.columns()));
        }
        c += maxWidths[column];
      }
      r += maxHeights[row];
    }

  }

  /**
   * Constructs a new diagonal matrix whose diagonal elements are the elements of <tt>vector</tt>. Cells values are
   * copied. The new matrix is not a view. Example:
   * <pre>
   * 5 4 3 -->
   * 5 0 0
   * 0 4 0
   * 0 0 3
   * </pre>
   *
   * @return a new matrix.
   */
  public ObjectMatrix2D diagonal(ObjectMatrix1D vector) {
    int size = vector.size();
    ObjectMatrix2D diag = make(size, size);
    for (int i = size; --i >= 0;) {
      diag.setQuick(i, i, vector.getQuick(i));
    }
    return diag;
  }

  /**
   * Constructs a new vector consisting of the diagonal elements of <tt>A</tt>. Cells values are copied. The new vector
   * is not a view. Example:
   * <pre>
   * 5 0 0 9
   * 0 4 0 9
   * 0 0 3 9
   * --> 5 4 3
   * </pre>
   *
   * @param A the matrix, need not be square.
   * @return a new vector.
   */
  public ObjectMatrix1D diagonal(ObjectMatrix2D A) {
    int min = Math.min(A.rows(), A.columns());
    ObjectMatrix1D diag = make1D(min);
    for (int i = min; --i >= 0;) {
      diag.setQuick(i, A.getQuick(i, i));
    }
    return diag;
  }

  /**
   * Constructs a matrix with the given cell values. <tt>values</tt> is required to have the form
   * <tt>values[row][column]</tt> and have exactly the same number of columns in every row. <p> The values are copied.
   * So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   * @throws IllegalArgumentException if <tt>for any 1 &lt;= row &lt; values.length: values[row].length !=
   *                                  values[row-1].length</tt>.
   */
  public ObjectMatrix2D make(Object[][] values) {
    if (this == sparse) {
      return new SparseObjectMatrix2D(values);
    } else {
      return new DenseObjectMatrix2D(values);
    }
  }

  /**
   * Construct a matrix from a one-dimensional column-major packed array, ala Fortran. Has the form
   * <tt>matrix.get(row,column) == values[row + column*rows]</tt>. The values are copied.
   *
   * @param values One-dimensional array of Objects, packed by columns (ala Fortran).
   * @param rows   the number of rows.
   * @throws IllegalArgumentException <tt>values.length</tt> must be a multiple of <tt>rows</tt>.
   */
  public ObjectMatrix2D make(Object[] values, int rows) {
    int columns = (rows != 0 ? values.length / rows : 0);
    if (rows * columns != values.length) {
      throw new IllegalArgumentException("Array length must be a multiple of m.");
    }

    ObjectMatrix2D matrix = make(rows, columns);
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        matrix.setQuick(row, column, values[row + column * rows]);
      }
    }
    return matrix;
  }

  /** Constructs a matrix with the given shape, each cell initialized with zero. */
  public ObjectMatrix2D make(int rows, int columns) {
    if (this == sparse) {
      return new SparseObjectMatrix2D(rows, columns);
    } else {
      return new DenseObjectMatrix2D(rows, columns);
    }
  }

  /** Constructs a matrix with the given shape, each cell initialized with the given value. */
  public ObjectMatrix2D make(int rows, int columns, Object initialValue) {
    if (initialValue == null) {
      return make(rows, columns);
    }
    return make(rows, columns).assign(initialValue);
  }

  /** Constructs a 1d matrix of the right dynamic type. */
  protected ObjectMatrix1D make1D(int size) {
    return make(0, 0).like1D(size);
  }

  /**
   * C = A||A||..||A; Constructs a new matrix which is duplicated both along the row and column dimension. Example:
   * <pre>
   * 0 1
   * 2 3
   * repeat(2,3) -->
   * 0 1 0 1 0 1
   * 2 3 2 3 2 3
   * 0 1 0 1 0 1
   * 2 3 2 3 2 3
   * </pre>
   */
  public ObjectMatrix2D repeat(ObjectMatrix2D A, int rowRepeat, int columnRepeat) {
    int r = A.rows();
    int c = A.columns();
    ObjectMatrix2D matrix = make(r * rowRepeat, c * columnRepeat);
    for (int i = rowRepeat; --i >= 0;) {
      for (int j = columnRepeat; --j >= 0;) {
        matrix.viewPart(r * i, c * j, r, c).assign(A);
      }
    }
    return matrix;
  }
}
