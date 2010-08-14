/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.doublealgo;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.Swapper;
import org.apache.mahout.math.function.IntComparator;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.DoubleMatrix3D;
import org.apache.mahout.math.matrix.impl.AbstractFormatter;

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
      org.apache.mahout.math.Sorting.mergeSort(fromIndex, toIndex, c, swapper);
    }
  };

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected Sorting() {
  }

  /** Compare two values, one of which is assumed to be Double.NaN */
  private static int compareNaN(double a, double b) {
    if (a != a) {
      if (b != b) {
        return 0; // NaN equals NaN
      } else {
        return 1;
      } // e.g. NaN > 5
    }
    return -1; // e.g. 5 < NaN
  }

  protected void runSort(int[] a, int fromIndex, int toIndex, IntComparator c) {
    org.apache.mahout.math.Sorting.quickSort(a, fromIndex, toIndex, c);
  }

  protected void runSort(int fromIndex, int toIndex, IntComparator c, org.apache.mahout.math.Swapper swapper) {
    org.apache.mahout.math.Sorting.quickSort(fromIndex, toIndex, c, swapper);
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
  public DoubleMatrix1D sort(final DoubleMatrix1D vector) {
    int[] indexes = new int[vector.size()]; // row indexes to reorder instead of matrix itself
    for (int i = indexes.length; --i >= 0;) {
      indexes[i] = i;
    }

    IntComparator comp = new IntComparator() {
      public int compare(int a, int b) {
        double av = vector.getQuick(a);
        double bv = vector.getQuick(b);
        if (av != av || bv != bv) {
          return compareNaN(av, bv);
        } // swap NaNs to the end
        return av < bv ? -1 : (av == bv ? 0 : 1);
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
   * DoubleComparator comp = new DoubleComparator() {
   * &nbsp;&nbsp;&nbsp;public int compare(double a, double b) {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double as = Math.sin(a); double bs = Math.sin(b);
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
  public DoubleMatrix1D sort(final DoubleMatrix1D vector, final org.apache.mahout.math.function.DoubleComparator c) {
    int[] indexes = new int[vector.size()]; // row indexes to reorder instead of matrix itself
    for (int i = indexes.length; --i >= 0;) {
      indexes[i] = i;
    }

    IntComparator comp = new IntComparator() {
      public int compare(int a, int b) {
        return c.compare(vector.getQuick(a), vector.getQuick(b));
      }
    };

    runSort(indexes, 0, indexes.length, comp);

    return vector.viewSelection(indexes);
  }

  /**
   * Sorts the matrix rows into ascending order, according to the <i>natural ordering</i> of the matrix values in the
   * virtual column <tt>aggregates</tt>; Particularly efficient when comparing expensive aggregates, because aggregates
   * need not be recomputed time and again, as is the case for comparator based sorts. Essentially, this algorithm makes
   * expensive comparisons cheap. Normally each element of <tt>aggregates</tt> is a summary measure of a row. Speedup
   * over comparator based sorting = <tt>2*log(rows)</tt>, on average. For this operation, quicksort is usually faster.
   * <p> The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa. To sort ranges use sub-ranging views. To sort columns by rows, use dice views. To sort descending, use
   * flip views ... <p> <b>Example:</b> Each aggregate is the sum of a row <table border="1" cellspacing="0"> <tr
   * nowrap> <td valign="top"><tt>4 x 2 matrix: <br> 1, 1<br> 5, 4<br> 3, 0<br> 4, 4 <br> </tt></td> <td align="left"
   * valign="top"> <tt>aggregates=<br> 2<br> 9<br> 3<br> 8<br> ==></tt></td> <td valign="top"> <p><tt>4 x 2 matrix:<br>
   * 1, 1<br> 3, 0<br> 4, 4<br> 5, 4</tt><br> The matrix IS NOT SORTED.<br> The new VIEW IS SORTED.</p> </td> </tr>
   * </table>
   *
   * <table> <td class="PRE">
   * <pre>
   * // sort 10000 x 1000 matrix by sum of logarithms in a row (i.e. by geometric mean)
   * DoubleMatrix2D matrix = new DenseDoubleMatrix2D(10000,1000);
   * matrix.assign(new engine.MersenneTwister()); // initialized randomly
   * org.apache.mahout.math.function.Functions F = org.apache.mahout.math.function.Functions.functions; // alias for convenience
   *
   * // THE QUICK VERSION (takes some 3 secs)
   * // aggregates[i] = Sum(log(row));
   * double[] aggregates = new double[matrix.rows()];
   * for (int i = matrix.rows(); --i >= 0; ) aggregates[i] = matrix.viewRow(i).aggregate(F.plus, F.log);
   * DoubleMatrix2D sorted = quickSort(matrix,aggregates);
   *
   * // THE SLOW VERSION (takes some 90 secs)
   * DoubleMatrix1DComparator comparator = new DoubleMatrix1DComparator() {
   * &nbsp;&nbsp;&nbsp;public int compare(DoubleMatrix1D x, DoubleMatrix1D y) {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a = x.aggregate(F.plus,F.log);
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double b = y.aggregate(F.plus,F.log);
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return a < b ? -1 : a==b ? 0 : 1;
   * &nbsp;&nbsp;&nbsp;}
   * };
   * DoubleMatrix2D sorted = quickSort(matrix,comparator);
   * </pre>
   * </td> </table>
   *
   * @param matrix     the matrix to be sorted.
   * @param aggregates the values to sort on. (As a side effect, this array will also get sorted).
   * @return a new matrix view having rows sorted. <b>Note that the original matrix is left unaffected.</b>
   * @throws IndexOutOfBoundsException if <tt>aggregates.length != matrix.rows()</tt>.
   */
  public DoubleMatrix2D sort(DoubleMatrix2D matrix, final double[] aggregates) {
    int rows = matrix.rows();
    if (aggregates.length != rows) {
      throw new IndexOutOfBoundsException("aggregates.length != matrix.rows()");
    }

    // set up index reordering
    final int[] indexes = new int[rows];
    for (int i = rows; --i >= 0;) {
      indexes[i] = i;
    }

    // compares two aggregates at a time
    IntComparator comp = new IntComparator() {
      public int compare(int x, int y) {
        double a = aggregates[x];
        double b = aggregates[y];
        if (a != a || b != b) {
          return compareNaN(a, b);
        } // swap NaNs to the end
        return a < b ? -1 : (a == b) ? 0 : 1;
      }
    };
    // swaps aggregates and reorders indexes
    Swapper swapper = new Swapper() {
      public void swap(int x, int y) {
        int t1 = indexes[x];
        indexes[x] = indexes[y];
        indexes[y] = t1;
        double t2 = aggregates[x];
        aggregates[x] = aggregates[y];
        aggregates[y] = t2;
      }
    };

    // sort indexes and aggregates
    runSort(0, rows, comp, swapper);

    // view the matrix according to the reordered row indexes
    // take all columns in the original order
    return matrix.viewSelection(indexes, null);
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
  public DoubleMatrix2D sort(DoubleMatrix2D matrix, int column) {
    if (column < 0 || column >= matrix.columns()) {
      throw new IndexOutOfBoundsException("column=" + column + ", matrix=" + AbstractFormatter.shape(matrix));
    }

    int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder instead of matrix itself
    for (int i = rowIndexes.length; --i >= 0;) {
      rowIndexes[i] = i;
    }

    final DoubleMatrix1D col = matrix.viewColumn(column);
    IntComparator comp = new IntComparator() {
      public int compare(int a, int b) {
        double av = col.getQuick(a);
        double bv = col.getQuick(b);
        if (av != av || bv != bv) {
          return compareNaN(av, bv);
        } // swap NaNs to the end
        return av < bv ? -1 : (av == bv ? 0 : 1);
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
   * DoubleMatrix1DComparator comp = new DoubleMatrix1DComparator() {
   * &nbsp;&nbsp;&nbsp;public int compare(DoubleMatrix1D a, DoubleMatrix1D b) {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double as = a.zSum(); double bs = b.zSum();
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
  public DoubleMatrix2D sort(DoubleMatrix2D matrix, final DoubleMatrix1DComparator c) {
    int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder instead of matrix itself
    for (int i = rowIndexes.length; --i >= 0;) {
      rowIndexes[i] = i;
    }

    final DoubleMatrix1D[] views = new DoubleMatrix1D[matrix.rows()]; // precompute views for speed
    for (int i = views.length; --i >= 0;) {
      views[i] = matrix.viewRow(i);
    }

    IntComparator comp = new IntComparator() {
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
 Sorts the matrix rows into ascending order, according to the <i>natural ordering</i> of the values computed by applying the given aggregation function to each row;
 Particularly efficient when comparing expensive aggregates, because aggregates need not be recomputed time and again, as is the case for comparator based sorts.
 Essentially, this algorithm makes expensive comparisons cheap.
 Normally <tt>aggregates</tt> defines a summary measure of a row.
 Speedup over comparator based sorting = <tt>2*log(rows)</tt>, on average.
 <p>
 The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
 To sort ranges use sub-ranging views. To sort columns by rows, use dice views. To sort descending, use flip views ...
 <p>
 <b>Example:</b>
 Each aggregate is the sum of a row
 <table border="1" cellspacing="0">
 <tr nowrap>
 <td valign="top"><tt>4 x 2 matrix: <br>
 1, 1<br>
 5, 4<br>
 3, 0<br>
 4, 4 <br>
 </tt></td>
 <td align="left" valign="top">
 <tt>aggregates=<br>
 hep.aida.bin.BinFunctions1D.sum<br>
 ==></tt></td>
 <td valign="top">
 <p><tt>4 x 2 matrix:<br>
 1, 1<br>
 3, 0<br>
 4, 4<br>
 5, 4</tt><br>
 The matrix IS NOT SORTED.<br>
 The new VIEW IS SORTED.</p>
 </td>
 </tr>
 </table>

 <table>
 <td class="PRE">
 <pre>
 // sort 10000 x 1000 matrix by median or by sum of logarithms in a row (i.e. by geometric mean)
 DoubleMatrix2D matrix = new DenseDoubleMatrix2D(10000,1000);
 matrix.assign(new engine.MersenneTwister()); // initialized randomly
 org.apache.mahout.math.function.Functions F = org.apache.mahout.math.function.Functions.functions; // alias for convenience

 // THE QUICK VERSION (takes some 10 secs)
 DoubleMatrix2D sorted = quickSort(matrix,hep.aida.bin.BinFunctions1D.median);
 //DoubleMatrix2D sorted = quickSort(matrix,hep.aida.bin.BinFunctions1D.sumOfLogarithms);

 // THE SLOW VERSION (takes some 300 secs)
 DoubleMatrix1DComparator comparator = new DoubleMatrix1DComparator() {
 &nbsp;&nbsp;&nbsp;public int compare(DoubleMatrix1D x, DoubleMatrix1D y) {
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a = org.apache.mahout.math.matrix.doublealgo.Statistic.bin(x).median();
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double b = org.apache.mahout.math.matrix.doublealgo.Statistic.bin(y).median();
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// double a = x.aggregate(F.plus,F.log);
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// double b = y.aggregate(F.plus,F.log);
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return a < b ? -1 : a==b ? 0 : 1;
 &nbsp;&nbsp;&nbsp;}
 };
 DoubleMatrix2D sorted = quickSort(matrix,comparator);
 </pre>
 </td>
 </table>

 @param matrix the matrix to be sorted.
 @param aggregate the function to sort on; aggregates values in a row.
 @return a new matrix view having rows sorted.
 <b>Note that the original matrix is left unaffected.</b>

 public DoubleMatrix2D sort(DoubleMatrix2D matrix, hep.aida.bin.BinFunction1D aggregate) {
 // precompute aggregates over rows, as defined by "aggregate"

 // a bit clumsy, because Statistic.aggregate(...) is defined on columns, so we need to transpose views
 DoubleMatrix2D tmp = matrix.like(1,matrix.rows());
 hep.aida.bin.BinFunction1D[] func = {aggregate};
 Statistic.aggregate(matrix.viewDice(), func, tmp);
 double[] aggr = tmp.viewRow(0).toArray();
 return sort(matrix,aggr);
 }
 */
  /**
   * Sorts the matrix slices into ascending order, according to the <i>natural ordering</i> of the matrix values in the
   * given <tt>[row,column]</tt> position. The returned view is backed by this matrix, so changes in the returned view
   * are reflected in this matrix, and vice-versa. To sort ranges use sub-ranging views. To sort by other dimensions,
   * use dice views. To sort descending, use flip views ... <p> The algorithm compares two 2-d slices at a time,
   * determining whether one is smaller, equal or larger than the other. Comparison is based on the cell
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
  public DoubleMatrix3D sort(DoubleMatrix3D matrix, int row, int column) {
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

    final DoubleMatrix1D sliceView = matrix.viewRow(row).viewColumn(column);
    IntComparator comp = new IntComparator() {
      public int compare(int a, int b) {
        double av = sliceView.getQuick(a);
        double bv = sliceView.getQuick(b);
        if (av != av || bv != bv) {
          return compareNaN(av, bv);
        } // swap NaNs to the end
        return av < bv ? -1 : (av == bv ? 0 : 1);
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
   * two slices (2-d matrices) at a time, determining whether one is smaller, equal or larger than the other. To sort
   * ranges use sub-ranging views. To sort by other dimensions, use dice views. To sort descending, use flip views ...
   * <p> <b>Example:</b>
   * <pre>
   * // sort by sum of values in a slice
   * DoubleMatrix2DComparator comp = new DoubleMatrix2DComparator() {
   * &nbsp;&nbsp;&nbsp;public int compare(DoubleMatrix2D a, DoubleMatrix2D b) {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double as = a.zSum(); double bs = b.zSum();
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
  public DoubleMatrix3D sort(DoubleMatrix3D matrix, final DoubleMatrix2DComparator c) {
    int[] sliceIndexes = new int[matrix.slices()]; // indexes to reorder instead of matrix itself
    for (int i = sliceIndexes.length; --i >= 0;) {
      sliceIndexes[i] = i;
    }

    final DoubleMatrix2D[] views = new DoubleMatrix2D[matrix.slices()]; // precompute views for speed
    for (int i = views.length; --i >= 0;) {
      views[i] = matrix.viewSlice(i);
    }

    IntComparator comp = new IntComparator() {
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
