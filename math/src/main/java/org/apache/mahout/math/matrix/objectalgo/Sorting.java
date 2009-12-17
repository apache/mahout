/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.objectalgo;

import org.apache.mahout.math.GenericSorting;
import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.function.IntComparator;
import org.apache.mahout.math.matrix.ObjectMatrix1D;
import org.apache.mahout.math.matrix.ObjectMatrix2D;
import org.apache.mahout.math.matrix.ObjectMatrix3D;
import org.apache.mahout.math.matrix.impl.AbstractFormatter;

import java.util.Comparator;
/**
 Matrix quicksorts and mergesorts.
 Use idioms like <tt>Sorting.quickSort.sort(...)</tt> and <tt>Sorting.mergeSort.sort(...)</tt>.
 <p>
 This is another case demonstrating one primary goal of this library: Delivering easy to use, yet very efficient APIs.
 The sorts return convenient <i>sort views</i>.
 This enables the usage of algorithms which scale well with the problem size:
 For example, sorting a 1000000 x 10000 or a 1000000 x 100 x 100 matrix performs just as fast as sorting a 1000000 x 1 matrix.
 This is so, because internally the algorithms only move around integer indexes, they do not physically move around entire rows or slices.
 The original matrix is left unaffected.
 <p>
 The quicksort is a derivative of the JDK 1.2 V1.26 algorithms (which are, in turn, based on Bentley's and McIlroy's fine work).
 The mergesort is a derivative of the JAL algorithms, with optimisations taken from the JDK algorithms.
 Mergesort is <i>stable</i> (by definition), while quicksort is not.
 A stable sort is, for example, helpful, if matrices are sorted successively
 by multiple columns. It preserves the relative position of equal elements.

 @see org.apache.mahout.math.GenericSorting
 @see org.apache.mahout.math.Sorting
 @see java.util.Arrays

 @author wolfgang.hoschek@cern.ch
 @version 1.1, 25/May/2000
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Sorting extends PersistentObject {

  /** A prefabricated quicksort. */
  public static final Sorting quickSort = new Sorting(); // already has quicksort implemented

  /** A prefabricated mergesort. */
  public static final Sorting mergeSort = new Sorting() { // override quicksort with mergesort

    @Override
    protected void runSort(int[] a, int fromIndex, int toIndex, IntComparator c) {
      org.apache.mahout.math.Sorting.mergeSort(a, fromIndex, toIndex, c);
    }

    @Override
    protected void runSort(int fromIndex, int toIndex, IntComparator c, org.apache.mahout.math.Swapper swapper) {
      GenericSorting.mergeSort(fromIndex, toIndex, c, swapper);
    }
  };

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private Sorting() {
  }

  protected void runSort(int[] a, int fromIndex, int toIndex, IntComparator c) {
    org.apache.mahout.math.Sorting.quickSort(a, fromIndex, toIndex, c);
  }

  protected void runSort(int fromIndex, int toIndex, IntComparator c, org.apache.mahout.math.Swapper swapper) {
    GenericSorting.quickSort(fromIndex, toIndex, c, swapper);
  }

  /**
   * Sorts the vector into ascending order, according to the <i>natural ordering</i>. The returned view is backed by
   * this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. To sort ranges use
   * sub-ranging views. To sort descending, use flip views ... <p> <b>Example:</b> <table border="1" cellspacing="0">
   * <tr nowrap> <td valign="top"><tt> 7, 1, 3, 1<br> </tt></td> <td valign="top"> <p><tt> ==&gt; 1, 1, 3, 7<br> The
   * vector IS NOT SORTED.<br> The new VIEW IS SORTED.</tt></p> </td> </tr> </table>
   *
   * @param vector the vector to be sorted.
   * @return a new sorted vector (matrix) view. <b>Note that the original matrix is left unaffected.</b>
   */
  public ObjectMatrix1D sort(final ObjectMatrix1D vector) {
    int[] indexes = new int[vector.size()]; // row indexes to reorder instead of matrix itself
    for (int i = indexes.length; --i >= 0;) {
      indexes[i] = i;
    }

    IntComparator comp = new IntComparator() {
      @Override
      public int compare(int a, int b) {
        Comparable<Object> av = (Comparable<Object>) (vector.getQuick(a));
        Comparable<Object> bv = (Comparable<Object>) (vector.getQuick(b));
        int r = av.compareTo(bv);
        return r < 0 ? -1 : (r > 0 ? 1 : 0);
      }
    };

    runSort(indexes, 0, indexes.length, comp);

    return vector.viewSelection(indexes);
  }

  /**
   * Sorts the vector into ascending order, according to the order induced by the specified comparator. The returned
   * view is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. The
   * algorithm compares two cells at a time, determinining whether one is smaller, equal or larger than the other. To
   * sort ranges use sub-ranging views. To sort descending, use flip views ... <p> <b>Example:</b>
   * <pre>
   * // sort by sinus of cells
   * ObjectComparator comp = new ObjectComparator() {
   * &nbsp;&nbsp;&nbsp;public int compare(Object a, Object b) {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Object as = Math.sin(a); Object bs = Math.sin(b);
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return as < bs ? -1 : as == bs ? 0 : 1;
   * &nbsp;&nbsp;&nbsp;}
   * };
   * sorted = quickSort(vector,comp);
   * </pre>
   *
   * @param vector the vector to be sorted.
   * @param c      the comparator to determine the order.
   * @return a new matrix view sorted as specified. <b>Note that the original vector (matrix) is left unaffected.</b>
   */
  public ObjectMatrix1D sort(final ObjectMatrix1D vector, final Comparator<Object> c) {
    int[] indexes = new int[vector.size()]; // row indexes to reorder instead of matrix itself
    for (int i = indexes.length; --i >= 0;) {
      indexes[i] = i;
    }

    IntComparator comp = new IntComparator() {
      @Override
      public int compare(int a, int b) {
        return c.compare(vector.getQuick(a), vector.getQuick(b));
      }
    };

    runSort(indexes, 0, indexes.length, comp);

    return vector.viewSelection(indexes);
  }

  /**
   * Sorts the matrix rows into ascending order, according to the <i>natural ordering</i> of the matrix values in the
   * given column. The returned view is backed by this matrix, so changes in the returned view are reflected in this
   * matrix, and vice-versa. To sort ranges use sub-ranging views. To sort columns by rows, use dice views. To sort
   * descending, use flip views ... <p> <b>Example:</b> <table border="1" cellspacing="0"> <tr nowrap> <td
   * valign="top"><tt>4 x 2 matrix: <br> 7, 6<br> 5, 4<br> 3, 2<br> 1, 0 <br> </tt></td> <td align="left" valign="top">
   * <p><tt>column = 0;<br> view = quickSort(matrix,column);<br> log.info(view); </tt><tt><br> ==> </tt></p> </td> <td
   * valign="top"> <p><tt>4 x 2 matrix:<br> 1, 0<br> 3, 2<br> 5, 4<br> 7, 6</tt><br> The matrix IS NOT SORTED.<br> The
   * new VIEW IS SORTED.</p> </td> </tr> </table>
   *
   * @param matrix the matrix to be sorted.
   * @param column the index of the column inducing the order.
   * @return a new matrix view having rows sorted by the given column. <b>Note that the original matrix is left
   *         unaffected.</b>
   * @throws IndexOutOfBoundsException if <tt>column < 0 || column >= matrix.columns()</tt>.
   */
  public ObjectMatrix2D sort(ObjectMatrix2D matrix, int column) {
    if (column < 0 || column >= matrix.columns()) {
      throw new IndexOutOfBoundsException("column=" + column + ", matrix=" + AbstractFormatter.shape(matrix));
    }

    int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder instead of matrix itself
    for (int i = rowIndexes.length; --i >= 0;) {
      rowIndexes[i] = i;
    }

    final ObjectMatrix1D col = matrix.viewColumn(column);
    IntComparator comp = new IntComparator() {
      @Override
      public int compare(int a, int b) {
        Comparable<Object> av = (Comparable<Object>) (col.getQuick(a));
        Comparable<Object> bv = (Comparable<Object>) (col.getQuick(b));
        int r = av.compareTo(bv);
        return r < 0 ? -1 : (r > 0 ? 1 : 0);
      }
    };

    runSort(rowIndexes, 0, rowIndexes.length, comp);

    // view the matrix according to the reordered row indexes
    // take all columns in the original order
    return matrix.viewSelection(rowIndexes, null);
  }

  /**
   * Sorts the matrix rows according to the order induced by the specified comparator. The returned view is backed by
   * this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. The algorithm compares
   * two rows (1-d matrices) at a time, determinining whether one is smaller, equal or larger than the other. To sort
   * ranges use sub-ranging views. To sort columns by rows, use dice views. To sort descending, use flip views ... <p>
   * <b>Example:</b>
   * <pre>
   * // sort by sum of values in a row
   * ObjectMatrix1DComparator comp = new ObjectMatrix1DComparator() {
   * &nbsp;&nbsp;&nbsp;public int compare(ObjectMatrix1D a, ObjectMatrix1D b) {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Object as = a.zSum(); Object bs = b.zSum();
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return as < bs ? -1 : as == bs ? 0 : 1;
   * &nbsp;&nbsp;&nbsp;}
   * };
   * sorted = quickSort(matrix,comp);
   * </pre>
   *
   * @param matrix the matrix to be sorted.
   * @param c      the comparator to determine the order.
   * @return a new matrix view having rows sorted as specified. <b>Note that the original matrix is left
   *         unaffected.</b>
   */
  public ObjectMatrix2D sort(ObjectMatrix2D matrix, final ObjectMatrix1DComparator c) {
    int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder instead of matrix itself
    for (int i = rowIndexes.length; --i >= 0;) {
      rowIndexes[i] = i;
    }

    final ObjectMatrix1D[] views = new ObjectMatrix1D[matrix.rows()]; // precompute views for speed
    for (int i = views.length; --i >= 0;) {
      views[i] = matrix.viewRow(i);
    }

    IntComparator comp = new IntComparator() {
      @Override
      public int compare(int a, int b) {
        //return c.compare(matrix.viewRow(a), matrix.viewRow(b));
        return c.compare(views[a], views[b]);
      }
    };

    runSort(rowIndexes, 0, rowIndexes.length, comp);

    // view the matrix according to the reordered row indexes
    // take all columns in the original order
    return matrix.viewSelection(rowIndexes, null);
  }

  /**
   * Sorts the matrix slices into ascending order, according to the <i>natural ordering</i> of the matrix values in the
   * given <tt>[row,column]</tt> position. The returned view is backed by this matrix, so changes in the returned view
   * are reflected in this matrix, and vice-versa. To sort ranges use sub-ranging views. To sort by other dimensions,
   * use dice views. To sort descending, use flip views ... <p> The algorithm compares two 2-d slices at a time,
   * determinining whether one is smaller, equal or larger than the other. Comparison is based on the cell
   * <tt>[row,column]</tt> within a slice. Let <tt>A</tt> and <tt>B</tt> be two 2-d slices. Then we have the following
   * rules <ul> <li><tt>A &lt;  B iff A.get(row,column) &lt;  B.get(row,column)</tt> <li><tt>A == B iff
   * A.get(row,column) == B.get(row,column)</tt> <li><tt>A &gt;  B  iff A.get(row,column) &gt;  B.get(row,column)</tt>
   * </ul>
   *
   * @param matrix the matrix to be sorted.
   * @param row    the index of the row inducing the order.
   * @param column the index of the column inducing the order.
   * @return a new matrix view having slices sorted by the values of the slice view <tt>matrix.viewRow(row).viewColumn(column)</tt>.
   *         <b>Note that the original matrix is left unaffected.</b>
   * @throws IndexOutOfBoundsException if <tt>row < 0 || row >= matrix.rows() || column < 0 || column >=
   *                                   matrix.columns()</tt>.
   */
  public ObjectMatrix3D sort(ObjectMatrix3D matrix, int row, int column) {
    if (row < 0 || row >= matrix.rows()) {
      throw new IndexOutOfBoundsException("row=" + row + ", matrix=" + AbstractFormatter.shape(matrix));
    }
    if (column < 0 || column >= matrix.columns()) {
      throw new IndexOutOfBoundsException("column=" + column + ", matrix=" + AbstractFormatter.shape(matrix));
    }

    int[] sliceIndexes = new int[matrix.slices()]; // indexes to reorder instead of matrix itself
    for (int i = sliceIndexes.length; --i >= 0;) {
      sliceIndexes[i] = i;
    }

    final ObjectMatrix1D sliceView = matrix.viewRow(row).viewColumn(column);
    IntComparator comp = new IntComparator() {
      @Override
      public int compare(int a, int b) {
        Comparable<Object> av = (Comparable<Object>) (sliceView.getQuick(a));
        Comparable<Object> bv = (Comparable<Object>) (sliceView.getQuick(b));
        int r = av.compareTo(bv);
        return r < 0 ? -1 : (r > 0 ? 1 : 0);
      }
    };

    runSort(sliceIndexes, 0, sliceIndexes.length, comp);

    // view the matrix according to the reordered slice indexes
    // take all rows and columns in the original order
    return matrix.viewSelection(sliceIndexes, null, null);
  }

  /**
   * Sorts the matrix slices according to the order induced by the specified comparator. The returned view is backed by
   * this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. The algorithm compares
   * two slices (2-d matrices) at a time, determinining whether one is smaller, equal or larger than the other. To sort
   * ranges use sub-ranging views. To sort by other dimensions, use dice views. To sort descending, use flip views ...
   * <p> <b>Example:</b>
   * <pre>
   * // sort by sum of values in a slice
   * ObjectMatrix2DComparator comp = new ObjectMatrix2DComparator() {
   * &nbsp;&nbsp;&nbsp;public int compare(ObjectMatrix2D a, ObjectMatrix2D b) {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Object as = a.zSum(); Object bs = b.zSum();
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return as < bs ? -1 : as == bs ? 0 : 1;
   * &nbsp;&nbsp;&nbsp;}
   * };
   * sorted = quickSort(matrix,comp);
   * </pre>
   *
   * @param matrix the matrix to be sorted.
   * @param c      the comparator to determine the order.
   * @return a new matrix view having slices sorted as specified. <b>Note that the original matrix is left
   *         unaffected.</b>
   */
  public ObjectMatrix3D sort(ObjectMatrix3D matrix, final ObjectMatrix2DComparator c) {
    int[] sliceIndexes = new int[matrix.slices()]; // indexes to reorder instead of matrix itself
    for (int i = sliceIndexes.length; --i >= 0;) {
      sliceIndexes[i] = i;
    }

    final ObjectMatrix2D[] views = new ObjectMatrix2D[matrix.slices()]; // precompute views for speed
    for (int i = views.length; --i >= 0;) {
      views[i] = matrix.viewSlice(i);
    }

    IntComparator comp = new IntComparator() {
      @Override
      public int compare(int a, int b) {
        //return c.compare(matrix.viewSlice(a), matrix.viewSlice(b));
        return c.compare(views[a], views[b]);
      }
    };

    runSort(sliceIndexes, 0, sliceIndexes.length, comp);

    // view the matrix according to the reordered slice indexes
    // take all rows and columns in the original order
    return matrix.viewSelection(sliceIndexes, null, null);
  }
}
