/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.impl;

import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/**
 * 2-d matrix holding <tt>double</tt> elements; either a view wrapping another matrix or a matrix whose views are
 * wrappers.
 *
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 04/14/2000
 */
class WrapperDoubleMatrix2D extends DoubleMatrix2D {
  /*
   * The elements of the matrix.
   */
  private final DoubleMatrix2D content;

  /**
   * Constructs a matrix with a copy of the given values. <tt>values</tt> is required to have the form
   * <tt>values[row][column]</tt> and have exactly the same number of columns in every row. <p> The values are copied.
   * So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param newContent The values to be filled into the new matrix.
   * @throws IllegalArgumentException if <tt>for any 1 &lt;= row &lt; values.length: values[row].length !=
   *                                  values[row-1].length</tt>.
   */
  WrapperDoubleMatrix2D(DoubleMatrix2D newContent) {
    if (newContent != null) {
      setUp(newContent.rows(), newContent.columns());
    }
    this.content = newContent;
  }

  /**
   * Returns the content of this matrix if it is a wrapper; or <tt>this</tt> otherwise. Override this method in
   * wrappers.
   */
  @Override
  protected DoubleMatrix2D getContent() {
    return content;
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
    return content.getQuick(row, column);
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
    return content.like(rows, columns);
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
    return content.like1D(size);
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
    throw new UnsupportedOperationException(); // should never get called
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
    content.setQuick(row, column, value);
  }

  /**
   * Constructs and returns a new <i>slice view</i> representing the rows of the given column. The returned view is
   * backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. To obtain a
   * slice view on subranges, construct a sub-ranging view (<tt>viewPart(...)</tt>), then apply this method to the
   * sub-range view. <p> <b>Example:</b> <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br>
   * 4, 5, 6 </td> <td>viewColumn(0) ==></td> <td valign="top">Matrix1D of size 2:<br> 1, 4</td> </tr> </table>
   *
   * @param column the column to fix.
   * @return a new slice view.
   * @throws IndexOutOfBoundsException if <tt>column < 0 || column >= columns()</tt>.
   * @see #viewRow(int)
   */
  @Override
  public DoubleMatrix1D viewColumn(int column) {
    return viewDice().viewRow(column);
  }

  /**
   * Constructs and returns a new <i>flip view</i> along the column axis. What used to be column <tt>0</tt> is now
   * column <tt>columns()-1</tt>, ..., what used to be column <tt>columns()-1</tt> is now column <tt>0</tt>. The
   * returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa. <p> <b>Example:</b> <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4,
   * 5, 6 </td> <td>columnFlip ==></td> <td valign="top">2 x 3 matrix:<br> 3, 2, 1 <br> 6, 5, 4</td> <td>columnFlip
   * ==></td> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4, 5, 6 </td> </tr> </table>
   *
   * @return a new flip view.
   * @see #viewRowFlip()
   */
  @Override
  public DoubleMatrix2D viewColumnFlip() {
    if (columns == 0) {
      return this;
    }
    return new WrapperDoubleMatrix2D(WrapperDoubleMatrix2D.this) {
      @Override
      public double getQuick(int row, int column) {
        return content.get(row, columns - 1 - column);
      }

      @Override
      public void setQuick(int row, int column, double value) {
        content.set(row, columns - 1 - column, value);
      }
    };
  }

  /**
   * Constructs and returns a new <i>dice (transposition) view</i>; Swaps axes; example: 3 x 4 matrix --> 4 x 3 matrix.
   * The view has both dimensions exchanged; what used to be columns become rows, what used to be rows become columns.
   * In other words: <tt>view.get(row,column)==this.get(column,row)</tt>. This is a zero-copy transposition, taking
   * O(1), i.e. constant time. The returned view is backed by this matrix, so changes in the returned view are reflected
   * in this matrix, and vice-versa. Use idioms like <tt>result = viewDice(A).copy()</tt> to generate an independent
   * transposed matrix. <p> <b>Example:</b> <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2,
   * 3<br> 4, 5, 6 </td> <td>transpose ==></td> <td valign="top">3 x 2 matrix:<br> 1, 4 <br> 2, 5 <br> 3, 6</td>
   * <td>transpose ==></td> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4, 5, 6 </td> </tr> </table>
   *
   * @return a new dice view.
   */
  @Override
  public DoubleMatrix2D viewDice() {
    DoubleMatrix2D view = new WrapperDoubleMatrix2D(this) {
      @Override
      public double getQuick(int row, int column) {
        return content.get(column, row);
      }

      @Override
      public void setQuick(int row, int column, double value) {
        content.set(column, row, value);
      }
    };
    view.rows = columns;
    view.columns = rows;
    return view;
  }

  /**
   * Constructs and returns a new <i>sub-range view</i> that is a <tt>height x width</tt> sub matrix starting at
   * <tt>[row,column]</tt>.
   *
   * Operations on the returned view can only be applied to the restricted range. Any attempt to access coordinates not
   * contained in the view will throw an <tt>IndexOutOfBoundsException</tt>. <p> <b>Note that the view is really just a
   * range restriction:</b> The returned matrix is backed by this matrix, so changes in the returned matrix are
   * reflected in this matrix, and vice-versa. <p> The view contains the cells from <tt>[row,column]</tt> to
   * <tt>[row+height-1,column+width-1]</tt>, all inclusive. and has <tt>view.rows() == height; view.columns() ==
   * width;</tt>. A view's legal coordinates are again zero based, as usual. In other words, legal coordinates of the
   * view range from <tt>[0,0]</tt> to <tt>[view.rows()-1==height-1,view.columns()-1==width-1]</tt>. As usual, any
   * attempt to access a cell at a coordinate <tt>column&lt;0 || column&gt;=view.columns() || row&lt;0 ||
   * row&gt;=view.rows()</tt> will throw an <tt>IndexOutOfBoundsException</tt>.
   *
   * @param row    The index of the row-coordinate.
   * @param column The index of the column-coordinate.
   * @param height The height of the box.
   * @param width  The width of the box.
   * @return the new view.
   * @throws IndexOutOfBoundsException if <tt>column<0 || width<0 || column+width>columns() || row<0 || height<0 ||
   *                                   row+height>rows()</tt>
   */
  @Override
  public DoubleMatrix2D viewPart(final int row, final int column, int height, int width) {
    checkBox(row, column, height, width);
    DoubleMatrix2D view = new WrapperDoubleMatrix2D(this) {
      @Override
      public double getQuick(int i, int j) {
        return content.get(row + i, column + j);
      }

      @Override
      public void setQuick(int i, int j, double value) {
        content.set(row + i, column + j, value);
      }
    };
    view.rows = height;
    view.columns = width;

    return view;
  }

  /**
   * Constructs and returns a new <i>slice view</i> representing the columns of the given row. The returned view is
   * backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. To obtain a
   * slice view on subranges, construct a sub-ranging view (<tt>viewPart(...)</tt>), then apply this method to the
   * sub-range view. <p> <b>Example:</b> <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br>
   * 4, 5, 6 </td> <td>viewRow(0) ==></td> <td valign="top">Matrix1D of size 3:<br> 1, 2, 3</td> </tr> </table>
   *
   * @param row the row to fix.
   * @return a new slice view.
   * @throws IndexOutOfBoundsException if <tt>row < 0 || row >= rows()</tt>.
   * @see #viewColumn(int)
   */
  @Override
  public DoubleMatrix1D viewRow(int row) {
    checkRow(row);
    return new DelegateDoubleMatrix1D(this, row);
  }

  /**
   * Constructs and returns a new <i>flip view</i> along the row axis. What used to be row <tt>0</tt> is now row
   * <tt>rows()-1</tt>, ..., what used to be row <tt>rows()-1</tt> is now row <tt>0</tt>. The returned view is backed by
   * this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. <p> <b>Example:</b>
   * <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4, 5, 6 </td> <td>rowFlip ==></td>
   * <td valign="top">2 x 3 matrix:<br> 4, 5, 6 <br> 1, 2, 3</td> <td>rowFlip ==></td> <td valign="top">2 x 3 matrix:
   * <br> 1, 2, 3<br> 4, 5, 6 </td> </tr> </table>
   *
   * @return a new flip view.
   * @see #viewColumnFlip()
   */
  @Override
  public DoubleMatrix2D viewRowFlip() {
    if (rows == 0) {
      return this;
    }
    return new WrapperDoubleMatrix2D(WrapperDoubleMatrix2D.this) {
      @Override
      public double getQuick(int row, int column) {
        return content.get(rows - 1 - row, column);
      }

      @Override
      public void setQuick(int row, int column, double value) {
        content.set(rows - 1 - row, column, value);
      }
    };
  }

  /**
   * Constructs and returns a new <i>selection view</i> that is a matrix holding the indicated cells. There holds
   * <tt>view.rows() == rowIndexes.length, view.columns() == columnIndexes.length</tt> and <tt>view.get(i,j) ==
   * this.get(rowIndexes[i],columnIndexes[j])</tt>. Indexes can occur multiple times and can be in arbitrary order. <p>
   * <b>Example:</b>
   * <pre>
   * this = 2 x 3 matrix:
   * 1, 2, 3
   * 4, 5, 6
   * rowIndexes     = (0,1)
   * columnIndexes  = (1,0,1,0)
   * -->
   * view = 2 x 4 matrix:
   * 2, 1, 2, 1
   * 5, 4, 5, 4
   * </pre>
   * Note that modifying the index arguments after this call has returned has no effect on the view. The returned view
   * is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. <p> To
   * indicate "all" rows or "all columns", simply set the respective parameter
   *
   * @param rowIndexes    The rows of the cells that shall be visible in the new view. To indicate that <i>all</i> rows
   *                      shall be visible, simply set this parameter to <tt>null</tt>.
   * @param columnIndexes The columns of the cells that shall be visible in the new view. To indicate that <i>all</i>
   *                      columns shall be visible, simply set this parameter to <tt>null</tt>.
   * @return the new view.
   * @throws IndexOutOfBoundsException if <tt>!(0 <= rowIndexes[i] < rows())</tt> for any <tt>i=0..rowIndexes.length()-1</tt>.
   * @throws IndexOutOfBoundsException if <tt>!(0 <= columnIndexes[i] < columns())</tt> for any
   *                                   <tt>i=0..columnIndexes.length()-1</tt>.
   */
  @Override
  public DoubleMatrix2D viewSelection(int[] rowIndexes, int[] columnIndexes) {
    // check for "all"
    if (rowIndexes == null) {
      rowIndexes = new int[rows];
      for (int i = rows; --i >= 0;) {
        rowIndexes[i] = i;
      }
    }
    if (columnIndexes == null) {
      columnIndexes = new int[columns];
      for (int i = columns; --i >= 0;) {
        columnIndexes[i] = i;
      }
    }

    checkRowIndexes(rowIndexes);
    checkColumnIndexes(columnIndexes);
    final int[] rix = rowIndexes;
    final int[] cix = columnIndexes;

    DoubleMatrix2D view = new WrapperDoubleMatrix2D(this) {
      @Override
      public double getQuick(int i, int j) {
        return content.get(rix[i], cix[j]);
      }

      @Override
      public void setQuick(int i, int j, double value) {
        content.set(rix[i], cix[j], value);
      }
    };
    view.rows = rowIndexes.length;
    view.columns = columnIndexes.length;

    return view;
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
    throw new UnsupportedOperationException(); // should never be called
  }

}
