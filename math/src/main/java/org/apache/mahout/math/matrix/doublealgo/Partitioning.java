/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.doublealgo;

import org.apache.mahout.math.Swapper;
import org.apache.mahout.math.function.IntComparator;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Partitioning {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private Partitioning() {
  }

  /**
   * Same as {@link org.apache.mahout.math.Partitioning#partition(int[],int,int,int[],int,int,int[])} except that it
   * <i>synchronously</i> partitions the rows of the given matrix by the values of the given matrix column; This is
   * essentially the same as partitioning a list of composite objects by some instance variable; In other words, two
   * entire rows of the matrix are swapped, whenever two column values indicate so. <p> Let's say, a "row" is an
   * "object" (tuple, d-dimensional point). A "column" is the list of "object" values of a given variable (field,
   * dimension). A "matrix" is a list of "objects" (tuples, points). <p> Now, rows (objects, tuples) are partially
   * sorted according to their values in one given variable (dimension). Two entire rows of the matrix are swapped,
   * whenever two column values indicate so. <p> Note that arguments are not checked for validity. <p> <b>Example:</b>
   * <table border="1" cellspacing="0"> <tr nowrap> <td valign="top"><tt>8 x 3 matrix:<br> 23, 22, 21<br> 20, 19, 18<br>
   * 17, 16, 15<br> 14, 13, 12<br> 11, 10, 9<br> 8,  7,  6<br> 5,  4,  3<br> 2,  1,  0 </tt></td> <td align="left"
   * valign="top"> <p><tt>column = 0;<br> rowIndexes = {0,1,2,..,matrix.rows()-1}; rowFrom = 0;<br> rowTo =
   * matrix.rows()-1;<br> splitters = {5,10,12}<br> c = 0; <br> d = splitters.length-1;<br>
   * partition(matrix,rowIndexes,rowFrom,rowTo,column,splitters,c,d,splitIndexes);<br> ==><br> splitIndexes == {0, 2,
   * 3}<br> rowIndexes == {7, 6, 5, 4, 0, 1, 2, 3}</tt></p> </td> <td valign="top"> The matrix IS NOT REORDERED.<br>
   * Here is how it would look<br> like, if it would be reordered<br> accoring to <tt>rowIndexes</tt>.<br> <tt>8 x 3
   * matrix:<br> 2,  1,  0<br> 5,  4,  3<br> 8,  7,  6<br> 11, 10, 9<br> 23, 22, 21<br> 20, 19, 18<br> 17, 16, 15<br>
   * 14, 13, 12 </tt></td> </tr> </table>
   *
   * @param matrix       the matrix to be partitioned.
   * @param rowIndexes   the index of the i-th row; is modified by this method to reflect partitioned indexes.
   * @param rowFrom      the index of the first row (inclusive).
   * @param rowTo        the index of the last row (inclusive).
   * @param column       the index of the column to partition on.
   * @param splitters    the values at which the rows shall be split into intervals. Must be sorted ascending and must
   *                     not contain multiple identical values. These preconditions are not checked; be sure that they
   *                     are met.
   * @param splitFrom    the index of the first splitter element to be considered.
   * @param splitTo      the index of the last splitter element to be considered. The method considers the splitter
   *                     elements <tt>splitters[splitFrom] .. splitters[splitTo]</tt>.
   * @param splitIndexes a list into which this method fills the indexes of rows delimiting intervals. Upon return
   *                     <tt>splitIndexes[splitFrom..splitTo]</tt> will be set accordingly. Therefore, must satisfy
   *                     <tt>splitIndexes.length >= splitters.length</tt>.
   */
  public static void partition(DoubleMatrix2D matrix, int[] rowIndexes, int rowFrom, int rowTo, int column,
                               final double[] splitters, int splitFrom, int splitTo, int[] splitIndexes) {
    if (rowFrom < 0 || rowTo >= matrix.rows() || rowTo >= rowIndexes.length) {
      throw new IllegalArgumentException();
    }
    if (column < 0 || column >= matrix.columns()) {
      throw new IllegalArgumentException();
    }
    if (splitFrom < 0 || splitTo >= splitters.length) {
      throw new IllegalArgumentException();
    }
    if (splitIndexes.length < splitters.length) {
      throw new IllegalArgumentException();
    }

    // this one knows how to swap two row indexes (a,b)
    final int[] g = rowIndexes;
    Swapper swapper = new Swapper() {
      public void swap(int b, int c) {
        int tmp = g[b];
        g[b] = g[c];
        g[c] = tmp;
      }
    };

    // compare splitter[a] with columnView[rowIndexes[b]]
    final DoubleMatrix1D columnView = matrix.viewColumn(column);
    IntComparator comp = new IntComparator() {
      public int compare(int a, int b) {
        double av = splitters[a];
        double bv = columnView.getQuick(g[b]);
        return av < bv ? -1 : (av == bv ? 0 : 1);
      }
    };

    // compare columnView[rowIndexes[a]] with columnView[rowIndexes[b]]
    IntComparator comp2 = new IntComparator() {
      public int compare(int a, int b) {
        double av = columnView.getQuick(g[a]);
        double bv = columnView.getQuick(g[b]);
        return av < bv ? -1 : (av == bv ? 0 : 1);
      }
    };

    // compare splitter[a] with splitter[b]
    IntComparator comp3 = new IntComparator() {
      public int compare(int a, int b) {
        double av = splitters[a];
        double bv = splitters[b];
        return av < bv ? -1 : (av == bv ? 0 : 1);
      }
    };

    // generic partitioning does the main work of reordering row indexes
    org.apache.mahout.math.Partitioning.genericPartition(
        rowFrom, rowTo, splitFrom, splitTo, splitIndexes, comp, comp2, comp3, swapper);
  }

  /**
   * Same as {@link org.apache.mahout.math.Partitioning#partition(int[],int,int,int[],int,int,int[])} except that it
   * <i>synchronously</i> partitions the rows of the given matrix by the values of the given matrix column; This is
   * essentially the same as partitioning a list of composite objects by some instance variable; In other words, two
   * entire rows of the matrix are swapped, whenever two column values indicate so. <p> Let's say, a "row" is an
   * "object" (tuple, d-dimensional point). A "column" is the list of "object" values of a given variable (field,
   * dimension). A "matrix" is a list of "objects" (tuples, points). <p> Now, rows (objects, tuples) are partially
   * sorted according to their values in one given variable (dimension). Two entire rows of the matrix are swapped,
   * whenever two column values indicate so. <p> Note that arguments are not checked for validity. <p> <b>Example:</b>
   * <table border="1" cellspacing="0"> <tr nowrap> <td valign="top"><tt>8 x 3 matrix:<br> 23, 22, 21<br> 20, 19, 18<br>
   * 17, 16, 15<br> 14, 13, 12<br> 11, 10, 9<br> 8,  7,  6<br> 5,  4,  3<br> 2,  1,  0 </tt></td> <td align="left"
   * valign="top"> <tt>column = 0;<br> splitters = {5,10,12}<br> partition(matrix,column,splitters,splitIndexes);<br>
   * ==><br> splitIndexes == {0, 2, 3}</tt></p> </td> <td valign="top"> The matrix IS NOT REORDERED.<br> The new VIEW IS
   * REORDERED:<br> <tt>8 x 3 matrix:<br> 2,  1,  0<br> 5,  4,  3<br> 8,  7,  6<br> 11, 10, 9<br> 23, 22, 21<br> 20, 19,
   * 18<br> 17, 16, 15<br> 14, 13, 12 </tt></td> </tr> </table>
   *
   * @param matrix       the matrix to be partitioned.
   * @param column       the index of the column to partition on.
   * @param splitters    the values at which the rows shall be split into intervals. Must be sorted ascending and must
   *                     not contain multiple identical values. These preconditions are not checked; be sure that they
   *                     are met.
   * @param splitIndexes a list into which this method fills the indexes of rows delimiting intervals. Therefore, must
   *                     satisfy <tt>splitIndexes.length >= splitters.length</tt>.
   * @return a new matrix view having rows partitioned by the given column and splitters.
   */
  public static DoubleMatrix2D partition(DoubleMatrix2D matrix, int column, double[] splitters, int[] splitIndexes) {
    int rowTo = matrix.rows() - 1;
    int splitTo = splitters.length - 1;
    int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder instead of matrix itself
    for (int i = rowIndexes.length; --i >= 0;) {
      rowIndexes[i] = i;
    }

    int rowFrom = 0;
    int splitFrom = 0;
    partition(matrix, rowIndexes, rowFrom, rowTo, column, splitters, splitFrom, splitTo, splitIndexes);

    // take all columns in the original order
    int[] columnIndexes = new int[matrix.columns()];
    for (int i = columnIndexes.length; --i >= 0;) {
      columnIndexes[i] = i;
    }

    // view the matrix according to the reordered row indexes
    return matrix.viewSelection(rowIndexes, columnIndexes);
  }

}
