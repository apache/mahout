/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.colt.matrix.doublealgo;

import org.apache.mahout.colt.function.IntComparator;
import org.apache.mahout.colt.matrix.DoubleFactory2D;
import org.apache.mahout.colt.matrix.DoubleFactory3D;
import org.apache.mahout.colt.matrix.DoubleMatrix1D;
import org.apache.mahout.colt.matrix.DoubleMatrix2D;
import org.apache.mahout.colt.matrix.DoubleMatrix3D;
import org.apache.mahout.colt.matrix.impl.DenseDoubleMatrix1D;
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
 
@see org.apache.mahout.colt.GenericSorting
@see org.apache.mahout.colt.Sorting
@see java.util.Arrays

@author wolfgang.hoschek@cern.ch
@version 1.1, 25/May/2000
*/
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class Sorting extends org.apache.mahout.colt.PersistentObject {
	/**
	 * A prefabricated quicksort.
	 */
	public static final Sorting quickSort = new Sorting(); // already has quicksort implemented

	/**
	 * A prefabricated mergesort.
	 */
	public static final Sorting mergeSort = new Sorting() { // override quicksort with mergesort
		protected void runSort(int[] a, int fromIndex, int toIndex, IntComparator c) {
			org.apache.mahout.colt.Sorting.mergeSort(a,fromIndex,toIndex,c);
		}
		protected void runSort(int fromIndex, int toIndex, IntComparator c, org.apache.mahout.colt.Swapper swapper) {
			org.apache.mahout.colt.GenericSorting.mergeSort(fromIndex, toIndex, c, swapper);
		}
	};
/**
 * Makes this class non instantiable, but still let's others inherit from it.
 */
protected Sorting() {}
/**
 * Compare two values, one of which is assumed to be Double.NaN
 */
private static final int compareNaN(double a, double b) {
	if (a!=a) {
		if (b!=b) return 0; // NaN equals NaN
		else return 1; // e.g. NaN > 5
	}
	return -1; // e.g. 5 < NaN
}
protected void runSort(int[] a, int fromIndex, int toIndex, IntComparator c) {
	org.apache.mahout.colt.Sorting.quickSort(a,fromIndex,toIndex,c);
}
protected void runSort(int fromIndex, int toIndex, IntComparator c, org.apache.mahout.colt.Swapper swapper) {
	org.apache.mahout.colt.GenericSorting.quickSort(fromIndex, toIndex, c, swapper);
}
/**
Sorts the vector into ascending order, according to the <i>natural ordering</i>.
The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
To sort ranges use sub-ranging views. To sort descending, use flip views ...
<p>
<b>Example:</b> 
<table border="1" cellspacing="0">
  <tr nowrap> 
	<td valign="top"><tt> 7, 1, 3, 1<br>
	  </tt></td>
	<td valign="top"> 
	  <p><tt> ==&gt; 1, 1, 3, 7<br>
		The vector IS NOT SORTED.<br>
		The new VIEW IS SORTED.</tt></p>
	</td>
  </tr>
</table>

@param vector the vector to be sorted.
@return a new sorted vector (matrix) view. 
		<b>Note that the original matrix is left unaffected.</b>
*/
public DoubleMatrix1D sort(final DoubleMatrix1D vector) {
	int[] indexes = new int[vector.size()]; // row indexes to reorder instead of matrix itself
	for (int i=indexes.length; --i >= 0; ) indexes[i] = i;

	IntComparator comp = new IntComparator() {  
		public int compare(int a, int b) {
			double av = vector.getQuick(a);
			double bv = vector.getQuick(b);
			if (av!=av || bv!=bv) return compareNaN(av,bv); // swap NaNs to the end
			return av<bv ? -1 : (av==bv ? 0 : 1);
		}
	};

	runSort(indexes,0,indexes.length,comp);

	return vector.viewSelection(indexes);
}
/**
Sorts the vector into ascending order, according to the order induced by the specified comparator.
The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
The algorithm compares two cells at a time, determinining whether one is smaller, equal or larger than the other.
To sort ranges use sub-ranging views. To sort descending, use flip views ...
<p>
<b>Example:</b>
<pre>
// sort by sinus of cells
DoubleComparator comp = new DoubleComparator() {
&nbsp;&nbsp;&nbsp;public int compare(double a, double b) {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double as = Math.sin(a); double bs = Math.sin(b);
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return as < bs ? -1 : as == bs ? 0 : 1;
&nbsp;&nbsp;&nbsp;}
};
sorted = quickSort(vector,comp);
</pre>

@param vector the vector to be sorted.
@param c the comparator to determine the order.
@return a new matrix view sorted as specified.
		<b>Note that the original vector (matrix) is left unaffected.</b>
*/
public DoubleMatrix1D sort(final DoubleMatrix1D vector, final org.apache.mahout.colt.function.DoubleComparator c) {
	int[] indexes = new int[vector.size()]; // row indexes to reorder instead of matrix itself
	for (int i=indexes.length; --i >= 0; ) indexes[i] = i;

	IntComparator comp = new IntComparator() {  
		public int compare(int a, int b) {
			return c.compare(vector.getQuick(a), vector.getQuick(b));
		}
	};

	runSort(indexes,0,indexes.length,comp);

	return vector.viewSelection(indexes);
}
/**
Sorts the matrix rows into ascending order, according to the <i>natural ordering</i> of the matrix values in the virtual column <tt>aggregates</tt>;
Particularly efficient when comparing expensive aggregates, because aggregates need not be recomputed time and again, as is the case for comparator based sorts.
Essentially, this algorithm makes expensive comparisons cheap.
Normally each element of <tt>aggregates</tt> is a summary measure of a row.
Speedup over comparator based sorting = <tt>2*log(rows)</tt>, on average.
For this operation, quicksort is usually faster.
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
	  2<br>
	  9<br>
	  3<br>
	  8<br>
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
// sort 10000 x 1000 matrix by sum of logarithms in a row (i.e. by geometric mean)
DoubleMatrix2D matrix = new DenseDoubleMatrix2D(10000,1000);
matrix.assign(new org.apache.mahout.jet.random.engine.MersenneTwister()); // initialized randomly
org.apache.mahout.jet.math.Functions F = org.apache.mahout.jet.math.Functions.functions; // alias for convenience

// THE QUICK VERSION (takes some 3 secs)
// aggregates[i] = Sum(log(row));
double[] aggregates = new double[matrix.rows()];
for (int i = matrix.rows(); --i >= 0; ) aggregates[i] = matrix.viewRow(i).aggregate(F.plus, F.log);
DoubleMatrix2D sorted = quickSort(matrix,aggregates);

// THE SLOW VERSION (takes some 90 secs)
DoubleMatrix1DComparator comparator = new DoubleMatrix1DComparator() {
&nbsp;&nbsp;&nbsp;public int compare(DoubleMatrix1D x, DoubleMatrix1D y) {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a = x.aggregate(F.plus,F.log);
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double b = y.aggregate(F.plus,F.log);
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return a < b ? -1 : a==b ? 0 : 1;
&nbsp;&nbsp;&nbsp;}
};
DoubleMatrix2D sorted = quickSort(matrix,comparator);
</pre>
</td>
</table>

@param matrix the matrix to be sorted.
@param aggregates the values to sort on. (As a side effect, this array will also get sorted).
@return a new matrix view having rows sorted.
		<b>Note that the original matrix is left unaffected.</b>
@throws IndexOutOfBoundsException if <tt>aggregates.length != matrix.rows()</tt>.
*/
public DoubleMatrix2D sort(DoubleMatrix2D matrix, final double[] aggregates) {
	int rows = matrix.rows();
	if (aggregates.length != rows) throw new IndexOutOfBoundsException("aggregates.length != matrix.rows()");
	
	// set up index reordering
	final int[] indexes = new int[rows];
	for (int i=rows; --i >= 0; ) indexes[i] = i;

	// compares two aggregates at a time
	org.apache.mahout.colt.function.IntComparator comp = new org.apache.mahout.colt.function.IntComparator() {
		public int compare(int x, int y) {
			double a = aggregates[x];
			double b = aggregates[y];
			if (a!=a || b!=b) return compareNaN(a,b); // swap NaNs to the end
			return a < b ? -1 : (a==b) ? 0 : 1;
		}
	};
	// swaps aggregates and reorders indexes
	org.apache.mahout.colt.Swapper swapper = new org.apache.mahout.colt.Swapper() {
		public void swap(int x, int y) {
			int t1;	double t2;
			t1 = indexes[x]; indexes[x] = indexes[y]; indexes[y] = t1;		
			t2 = aggregates[x]; aggregates[x] = aggregates[y]; aggregates[y] = t2;
		}
	};

	// sort indexes and aggregates
	runSort(0,rows,comp,swapper);

	// view the matrix according to the reordered row indexes
	// take all columns in the original order
	return matrix.viewSelection(indexes,null);
}
/**
Sorts the matrix rows into ascending order, according to the <i>natural ordering</i> of the matrix values in the given column.
The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
To sort ranges use sub-ranging views. To sort columns by rows, use dice views. To sort descending, use flip views ...
<p>
<b>Example:</b> 
<table border="1" cellspacing="0">
  <tr nowrap> 
	<td valign="top"><tt>4 x 2 matrix: <br>
	  7, 6<br>
	  5, 4<br>
	  3, 2<br>
	  1, 0 <br>
	  </tt></td>
	<td align="left" valign="top"> 
	  <p><tt>column = 0;<br>
		view = quickSort(matrix,column);<br>
		System.out.println(view); </tt><tt><br>
		==> </tt></p>
	  </td>
	<td valign="top"> 
	  <p><tt>4 x 2 matrix:<br>
		1, 0<br>
		3, 2<br>
		5, 4<br>
		7, 6</tt><br>
		The matrix IS NOT SORTED.<br>
		The new VIEW IS SORTED.</p>
	  </td>
  </tr>
</table>

@param matrix the matrix to be sorted.
@param column the index of the column inducing the order.
@return a new matrix view having rows sorted by the given column.
		<b>Note that the original matrix is left unaffected.</b>
@throws IndexOutOfBoundsException if <tt>column < 0 || column >= matrix.columns()</tt>.
*/
public DoubleMatrix2D sort(DoubleMatrix2D matrix, int column) {
	if (column < 0 || column >= matrix.columns()) throw new IndexOutOfBoundsException("column="+column+", matrix="+Formatter.shape(matrix));

	int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder instead of matrix itself
	for (int i=rowIndexes.length; --i >= 0; ) rowIndexes[i] = i;

	final DoubleMatrix1D col = matrix.viewColumn(column);
	IntComparator comp = new IntComparator() {  
		public int compare(int a, int b) {
			double av = col.getQuick(a);
			double bv = col.getQuick(b);
			if (av!=av || bv!=bv) return compareNaN(av,bv); // swap NaNs to the end
			return av<bv ? -1 : (av==bv ? 0 : 1);
		}
	};

	runSort(rowIndexes,0,rowIndexes.length,comp);

	// view the matrix according to the reordered row indexes
	// take all columns in the original order
	return matrix.viewSelection(rowIndexes,null);
}
/**
Sorts the matrix rows according to the order induced by the specified comparator.
The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
The algorithm compares two rows (1-d matrices) at a time, determinining whether one is smaller, equal or larger than the other.
To sort ranges use sub-ranging views. To sort columns by rows, use dice views. To sort descending, use flip views ...
<p>
<b>Example:</b>
<pre>
// sort by sum of values in a row
DoubleMatrix1DComparator comp = new DoubleMatrix1DComparator() {
&nbsp;&nbsp;&nbsp;public int compare(DoubleMatrix1D a, DoubleMatrix1D b) {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double as = a.zSum(); double bs = b.zSum();
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return as < bs ? -1 : as == bs ? 0 : 1;
&nbsp;&nbsp;&nbsp;}
};
sorted = quickSort(matrix,comp);
</pre>

@param matrix the matrix to be sorted.
@param c the comparator to determine the order.
@return a new matrix view having rows sorted as specified.
		<b>Note that the original matrix is left unaffected.</b>
*/
public DoubleMatrix2D sort(final DoubleMatrix2D matrix, final DoubleMatrix1DComparator c) {
	int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder instead of matrix itself
	for (int i=rowIndexes.length; --i >= 0; ) rowIndexes[i] = i;

	final DoubleMatrix1D[] views = new DoubleMatrix1D[matrix.rows()]; // precompute views for speed
	for (int i=views.length; --i >= 0; ) views[i] = matrix.viewRow(i);

	IntComparator comp = new IntComparator() {  
		public int compare(int a, int b) {
			//return c.compare(matrix.viewRow(a), matrix.viewRow(b));
			return c.compare(views[a], views[b]);
		}
	};

	runSort(rowIndexes,0,rowIndexes.length,comp);

	// view the matrix according to the reordered row indexes
	// take all columns in the original order
	return matrix.viewSelection(rowIndexes,null);
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
matrix.assign(new org.apache.mahout.jet.random.engine.MersenneTwister()); // initialized randomly
org.apache.mahout.jet.math.Functions F = org.apache.mahout.jet.math.Functions.functions; // alias for convenience

// THE QUICK VERSION (takes some 10 secs)
DoubleMatrix2D sorted = quickSort(matrix,hep.aida.bin.BinFunctions1D.median);
//DoubleMatrix2D sorted = quickSort(matrix,hep.aida.bin.BinFunctions1D.sumOfLogarithms);

// THE SLOW VERSION (takes some 300 secs)
DoubleMatrix1DComparator comparator = new DoubleMatrix1DComparator() {
&nbsp;&nbsp;&nbsp;public int compare(DoubleMatrix1D x, DoubleMatrix1D y) {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a = org.apache.mahout.colt.matrix.doublealgo.Statistic.bin(x).median();
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double b = org.apache.mahout.colt.matrix.doublealgo.Statistic.bin(y).median();
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
Sorts the matrix slices into ascending order, according to the <i>natural ordering</i> of the matrix values in the given <tt>[row,column]</tt> position.
The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
To sort ranges use sub-ranging views. To sort by other dimensions, use dice views. To sort descending, use flip views ...
<p>
The algorithm compares two 2-d slices at a time, determinining whether one is smaller, equal or larger than the other.
Comparison is based on the cell <tt>[row,column]</tt> within a slice.
Let <tt>A</tt> and <tt>B</tt> be two 2-d slices. Then we have the following rules
<ul>
<li><tt>A &lt;  B  iff A.get(row,column) &lt;  B.get(row,column)</tt>
<li><tt>A == B iff A.get(row,column) == B.get(row,column)</tt>
<li><tt>A &gt;  B  iff A.get(row,column) &gt;  B.get(row,column)</tt>
</ul>

@param matrix the matrix to be sorted.
@param row the index of the row inducing the order.
@param column the index of the column inducing the order.
@return a new matrix view having slices sorted by the values of the slice view <tt>matrix.viewRow(row).viewColumn(column)</tt>.
		<b>Note that the original matrix is left unaffected.</b>
@throws IndexOutOfBoundsException if <tt>row < 0 || row >= matrix.rows() || column < 0 || column >= matrix.columns()</tt>.
*/
public DoubleMatrix3D sort(DoubleMatrix3D matrix, int row, int column) {
	if (row < 0 || row >= matrix.rows()) throw new IndexOutOfBoundsException("row="+row+", matrix="+Formatter.shape(matrix));
	if (column < 0 || column >= matrix.columns()) throw new IndexOutOfBoundsException("column="+column+", matrix="+Formatter.shape(matrix));

	int[] sliceIndexes = new int[matrix.slices()]; // indexes to reorder instead of matrix itself
	for (int i=sliceIndexes.length; --i >= 0; ) sliceIndexes[i] = i;

	final DoubleMatrix1D sliceView = matrix.viewRow(row).viewColumn(column);
	IntComparator comp = new IntComparator() {  
		public int compare(int a, int b) {
			double av = sliceView.getQuick(a);
			double bv = sliceView.getQuick(b);
			if (av!=av || bv!=bv) return compareNaN(av,bv); // swap NaNs to the end
			return av<bv ? -1 : (av==bv ? 0 : 1);
		}
	};

	runSort(sliceIndexes,0,sliceIndexes.length,comp);

	// view the matrix according to the reordered slice indexes
	// take all rows and columns in the original order
	return matrix.viewSelection(sliceIndexes,null,null);
}
/**
Sorts the matrix slices according to the order induced by the specified comparator.
The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
The algorithm compares two slices (2-d matrices) at a time, determinining whether one is smaller, equal or larger than the other.
To sort ranges use sub-ranging views. To sort by other dimensions, use dice views. To sort descending, use flip views ...
<p>
<b>Example:</b>
<pre>
// sort by sum of values in a slice
DoubleMatrix2DComparator comp = new DoubleMatrix2DComparator() {
&nbsp;&nbsp;&nbsp;public int compare(DoubleMatrix2D a, DoubleMatrix2D b) {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double as = a.zSum(); double bs = b.zSum();
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return as < bs ? -1 : as == bs ? 0 : 1;
&nbsp;&nbsp;&nbsp;}
};
sorted = quickSort(matrix,comp);
</pre>

@param matrix the matrix to be sorted.
@param c the comparator to determine the order.
@return a new matrix view having slices sorted as specified.
		<b>Note that the original matrix is left unaffected.</b>
*/
public DoubleMatrix3D sort(final DoubleMatrix3D matrix, final DoubleMatrix2DComparator c) {
	int[] sliceIndexes = new int[matrix.slices()]; // indexes to reorder instead of matrix itself
	for (int i=sliceIndexes.length; --i >= 0; ) sliceIndexes[i] = i;

	final DoubleMatrix2D[] views = new DoubleMatrix2D[matrix.slices()]; // precompute views for speed
	for (int i=views.length; --i >= 0; ) views[i] = matrix.viewSlice(i);

	IntComparator comp = new IntComparator() {  
		public int compare(int a, int b) {
			//return c.compare(matrix.viewSlice(a), matrix.viewSlice(b));
			return c.compare(views[a], views[b]);
		}
	};

	runSort(sliceIndexes,0,sliceIndexes.length,comp);

	// view the matrix according to the reordered slice indexes
	// take all rows and columns in the original order
	return matrix.viewSelection(sliceIndexes,null,null);
}
/**
 * Demonstrates advanced sorting.
 * Sorts by sum of row.
 */
public static void zdemo1() {
	Sorting sort = quickSort;
	DoubleMatrix2D matrix = DoubleFactory2D.dense.descending(4,3);
	DoubleMatrix1DComparator comp = new DoubleMatrix1DComparator() {
		public int compare(DoubleMatrix1D a, DoubleMatrix1D b) {
			double as = a.zSum(); double bs = b.zSum();
			return as < bs ? -1 : as == bs ? 0 : 1;
		}
	};
	System.out.println("unsorted:"+matrix);
	System.out.println("sorted  :"+sort.sort(matrix,comp));
}
/**
 * Demonstrates advanced sorting.
 * Sorts by sum of slice.
 */
public static void zdemo2() {
	Sorting sort = quickSort;
	DoubleMatrix3D matrix = DoubleFactory3D.dense.descending(4,3,2);
	DoubleMatrix2DComparator comp = new DoubleMatrix2DComparator() {
		public int compare(DoubleMatrix2D a, DoubleMatrix2D b) {
			double as = a.zSum();
			double bs = b.zSum();
			return as < bs ? -1 : as == bs ? 0 : 1;
		}
	};
	System.out.println("unsorted:"+matrix);
	System.out.println("sorted  :"+sort.sort(matrix,comp));
}
/**
 * Demonstrates advanced sorting.
 * Sorts by sinus of cell values.
 */
public static void zdemo3() {
	Sorting sort = quickSort;
	double[] values = {0.5, 1.5, 2.5, 3.5};
	DoubleMatrix1D matrix = new DenseDoubleMatrix1D(values);
	org.apache.mahout.colt.function.DoubleComparator comp = new org.apache.mahout.colt.function.DoubleComparator() {
		public int compare(double a, double b) {
			double as = Math.sin(a); double bs = Math.sin(b);
			return as < bs ? -1 : as == bs ? 0 : 1;
		}
	};
	System.out.println("unsorted:"+matrix);
	
	DoubleMatrix1D sorted = sort.sort(matrix,comp);
	System.out.println("sorted  :"+sorted);

	// check whether it is really sorted
	sorted.assign(org.apache.mahout.jet.math.Functions.sin);
	/*
	sorted.assign(
		new org.apache.mahout.colt.function.DoubleFunction() {
			public double apply(double arg) { return Math.sin(arg); }
		}
	);
	*/
	System.out.println("sined  :"+sorted);
}
/**
 * Demonstrates applying functions.
 */
protected static void zdemo4() {
	double[] values1 = {0, 1, 2, 3};
	double[] values2 = {0, 2, 4, 6};
	DoubleMatrix1D matrix1 = new DenseDoubleMatrix1D(values1);
	DoubleMatrix1D matrix2 = new DenseDoubleMatrix1D(values2);
	System.out.println("m1:"+matrix1);
	System.out.println("m2:"+matrix2);
	
	matrix1.assign(matrix2, org.apache.mahout.jet.math.Functions.pow);
	
	/*
	matrix1.assign(matrix2,
		new org.apache.mahout.colt.function.DoubleDoubleFunction() {
			public double apply(double x, double y) { return Math.pow(x,y); }
		}
	);
	*/
	
	System.out.println("applied:"+matrix1);
}
/**
 * Demonstrates sorting with precomputation of aggregates (median and sum of logarithms).
 *
public static void zdemo5(int rows, int columns, boolean print) {
	Sorting sort = quickSort;
	// for reliable benchmarks, call this method twice: once with small dummy parameters to "warm up" the jitter, then with your real work-load
		
	System.out.println("\n\n");
	System.out.print("now initializing... ");
	org.apache.mahout.colt.Timer timer = new org.apache.mahout.colt.Timer().start();
		
	final org.apache.mahout.jet.math.Functions F = org.apache.mahout.jet.math.Functions.functions;
	DoubleMatrix2D A = org.apache.mahout.colt.matrix.DoubleFactory2D.dense.make(rows,columns);
	A.assign(new org.apache.mahout.jet.random.engine.DRand()); // initialize randomly
	timer.stop().display();

	// also benchmark copying in its several implementation flavours
	DoubleMatrix2D B = A.like();
	timer.reset().start();
	System.out.print("now copying... ");
	B.assign(A);
	timer.stop().display();

	timer.reset().start();
	System.out.print("now copying subrange... ");
	B.viewPart(0,0,rows,columns).assign(A.viewPart(0,0,rows,columns));
	timer.stop().display();
	//System.out.println(A);

	timer.reset().start();
	System.out.print("now copying selected... ");
	B.viewSelection(null,null).assign(A.viewSelection(null,null));
	timer.stop().display();

	System.out.print("now sorting - quick version with precomputation... ");
	timer.reset().start();
	// THE QUICK VERSION (takes some 10 secs)
	A = sort.sort(A,hep.aida.bin.BinFunctions1D.median);
	//A = sort.sort(A,hep.aida.bin.BinFunctions1D.sumLog);
	timer.stop().display();

	// check results for correctness
	// WARNING: be sure NOT TO PRINT huge matrices unless you have tons of main memory and time!!
	// so we just show the first 5 rows
	if (print) {
		int r = Math.min(rows,5);
		hep.aida.bin.BinFunction1D[] funs = {hep.aida.bin.BinFunctions1D.median, hep.aida.bin.BinFunctions1D.sumLog, hep.aida.bin.BinFunctions1D.geometricMean};
		String[] rowNames = new String[r];
		String[] columnNames = new String[columns];
		for (int i=columns; --i >= 0; ) columnNames[i] = Integer.toString(i);
		for (int i=r; --i >= 0; ) rowNames[i] = Integer.toString(i);
		System.out.println("first part of sorted result = \n"+new org.apache.mahout.colt.matrix.doublealgo.Formatter("%G").toTitleString(
			A.viewPart(0,0,r,columns), rowNames, columnNames, null, null, null, funs
		));
	}


	System.out.print("now sorting - slow version... ");
	A = B;	
	org.apache.mahout.colt.matrix.doublealgo.DoubleMatrix1DComparator fun = new org.apache.mahout.colt.matrix.doublealgo.DoubleMatrix1DComparator() {
		public int compare(DoubleMatrix1D x, DoubleMatrix1D y) {
			double a = org.apache.mahout.colt.matrix.doublealgo.Statistic.bin(x).median();
			double b = org.apache.mahout.colt.matrix.doublealgo.Statistic.bin(y).median();
			//double a = x.aggregate(F.plus,F.log);
			//double b = y.aggregate(F.plus,F.log);
			return a < b ? -1 : (a==b) ? 0 : 1;
		}
	};
	timer.reset().start();
	A = sort.sort(A,fun);
	timer.stop().display();
}
*/
/**
 * Demonstrates advanced sorting.
 * Sorts by sum of row.
 */
public static void zdemo6() {
	Sorting sort = quickSort;
	double[][] values = {
		{ 3,7,0 },
		{ 2,1,0 },
		{ 2,2,0 },
		{ 1,8,0 },
		{ 2,5,0 },
		{ 7,0,0 },
		{ 2,3,0 },
		{ 1,0,0 },
		{ 4,0,0 },
		{ 2,0,0 }
	};
	DoubleMatrix2D A = DoubleFactory2D.dense.make(values);
	DoubleMatrix2D B,C;
	/*
	DoubleMatrix1DComparator comp = new DoubleMatrix1DComparator() {
		public int compare(DoubleMatrix1D a, DoubleMatrix1D b) {
			double as = a.zSum(); double bs = b.zSum();
			return as < bs ? -1 : as == bs ? 0 : 1;
		}
	};
	*/
	System.out.println("\n\nunsorted:"+A);
	B = quickSort.sort(A,1);
	C = quickSort.sort(B,0);	
	System.out.println("quick sorted  :"+C);
	
	B = mergeSort.sort(A,1);
	C = mergeSort.sort(B,0);	
	System.out.println("merge sorted  :"+C);
	
}
/**
 * Demonstrates sorting with precomputation of aggregates, comparing mergesort with quicksort.
 */
public static void zdemo7(int rows, int columns, boolean print) {
	// for reliable benchmarks, call this method twice: once with small dummy parameters to "warm up" the jitter, then with your real work-load
		
	System.out.println("\n\n");
	System.out.println("now initializing... ");
		
	final org.apache.mahout.jet.math.Functions F = org.apache.mahout.jet.math.Functions.functions;
	DoubleMatrix2D A = org.apache.mahout.colt.matrix.DoubleFactory2D.dense.make(rows,columns);
	A.assign(new org.apache.mahout.jet.random.engine.DRand()); // initialize randomly
	DoubleMatrix2D B = A.copy();

	double[] v1 = A.viewColumn(0).toArray();
	double[] v2 = A.viewColumn(0).toArray();
	System.out.print("now quick sorting... ");
	org.apache.mahout.colt.Timer timer = new org.apache.mahout.colt.Timer().start();
	quickSort.sort(A,0);
	timer.stop().display();

	System.out.print("now merge sorting... ");
	timer.reset().start();
	mergeSort.sort(A,0);
	timer.stop().display();

	System.out.print("now quick sorting with simple aggregation... ");
	timer.reset().start();
	quickSort.sort(A,v1);
	timer.stop().display();

	System.out.print("now merge sorting with simple aggregation... ");
	timer.reset().start();
	mergeSort.sort(A,v2);
	timer.stop().display();
}
}
