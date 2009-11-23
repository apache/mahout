/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.doublealgo;

import org.apache.mahout.matrix.Swapper;
import org.apache.mahout.matrix.function.IntComparator;
import org.apache.mahout.matrix.matrix.DoubleMatrix1D;
import org.apache.mahout.matrix.matrix.DoubleMatrix2D;
/**
 * Given some interval boundaries, partitions matrices such that cell values falling into an interval are placed next to each other.
 * <p>
 * <b>Performance</b>
 * <p>
 * Partitioning into two intervals is <tt>O( N )</tt>.
 * Partitioning into k intervals is <tt>O( N * log(k))</tt>.
 * Constants factors are minimized. 
 *
 * @see org.apache.mahout.matrix.Partitioning "Partitioning arrays (provides more documentation)"
 *
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class Partitioning extends Object {
/**
 * Makes this class non instantiable, but still let's others inherit from it.
 */
protected Partitioning() {}
/**
Same as {@link org.apache.mahout.matrix.Partitioning#partition(int[],int,int,int[],int,int,int[])}
except that it <i>synchronously</i> partitions the rows of the given matrix by the values of the given matrix column;
This is essentially the same as partitioning a list of composite objects by some instance variable;
In other words, two entire rows of the matrix are swapped, whenever two column values indicate so.
<p>
Let's say, a "row" is an "object" (tuple, d-dimensional point).
A "column" is the list of "object" values of a given variable (field, dimension).
A "matrix" is a list of "objects" (tuples, points).
<p>
Now, rows (objects, tuples) are partially sorted according to their values in one given variable (dimension).
Two entire rows of the matrix are swapped, whenever two column values indicate so.
<p>
Note that arguments are not checked for validity.
<p>
<b>Example:</b> 
<table border="1" cellspacing="0">
  <tr nowrap> 
	<td valign="top"><tt>8 x 3 matrix:<br>
	  23, 22, 21<br>
	  20, 19, 18<br>
	  17, 16, 15<br>
	  14, 13, 12<br>
	  11, 10, 9<br>
	  8,  7,  6<br>
	  5,  4,  3<br>
	  2,  1,  0 </tt></td>
	<td align="left" valign="top"> 
	  <p><tt>column = 0;<br>
	    rowIndexes = {0,1,2,..,matrix.rows()-1};
		rowFrom = 0;<br>
		rowTo = matrix.rows()-1;<br>
		splitters = {5,10,12}<br>
		c = 0; <br>
		d = splitters.length-1;<br>
		partition(matrix,rowIndexes,rowFrom,rowTo,column,splitters,c,d,splitIndexes);<br>
		==><br>
		splitIndexes == {0, 2, 3}<br>
		rowIndexes == {7, 6, 5, 4, 0, 1, 2, 3}</tt></p>
	  </td>
	<td valign="top">
	  The matrix IS NOT REORDERED.<br>
	  Here is how it would look<br>
	  like, if it would be reordered<br>
	  accoring to <tt>rowIndexes</tt>.<br>
	  <tt>8 x 3 matrix:<br>
	  2,  1,  0<br>
	  5,  4,  3<br>
	  8,  7,  6<br>
	  11, 10, 9<br>
	  23, 22, 21<br>
	  20, 19, 18<br>
	  17, 16, 15<br>
	  14, 13, 12 </tt></td>
  </tr>
</table>
@param matrix the matrix to be partitioned.
@param rowIndexes the index of the i-th row; is modified by this method to reflect partitioned indexes.
@param rowFrom the index of the first row (inclusive).
@param rowTo the index of the last row (inclusive).
@param column the index of the column to partition on.
@param splitters the values at which the rows shall be split into intervals.
	Must be sorted ascending and must not contain multiple identical values.
	These preconditions are not checked; be sure that they are met.
 
@param splitFrom the index of the first splitter element to be considered.
@param splitTo the index of the last splitter element to be considered.
	The method considers the splitter elements <tt>splitters[splitFrom] .. splitters[splitTo]</tt>.
 
@param splitIndexes a list into which this method fills the indexes of rows delimiting intervals.
Upon return <tt>splitIndexes[splitFrom..splitTo]</tt> will be set accordingly.
Therefore, must satisfy <tt>splitIndexes.length >= splitters.length</tt>.
*/
public static void partition(DoubleMatrix2D matrix, int[] rowIndexes, int rowFrom, int rowTo, int column, final double[] splitters, int splitFrom, int splitTo, int[] splitIndexes) {
	if (rowFrom < 0 || rowTo >= matrix.rows() || rowTo >= rowIndexes.length) throw new IllegalArgumentException();
	if (column < 0 || column >= matrix.columns()) throw new IllegalArgumentException();
	if (splitFrom < 0 || splitTo >= splitters.length) throw new IllegalArgumentException();
	if (splitIndexes.length < splitters.length) throw new IllegalArgumentException();

	// this one knows how to swap two row indexes (a,b)
	final int[] g = rowIndexes;
	Swapper swapper = new Swapper() {
		public void swap(int b, int c) {
			int tmp = g[b]; g[b] = g[c]; g[c] = tmp;
		}
	};
	
	// compare splitter[a] with columnView[rowIndexes[b]]
	final DoubleMatrix1D columnView = matrix.viewColumn(column);	
	IntComparator comp = new IntComparator() {
		public int compare(int a, int b) {
			double av = splitters[a];
			double bv = columnView.getQuick(g[b]);
			return av<bv ? -1 : (av==bv ? 0 : 1);
		}
	};

	// compare columnView[rowIndexes[a]] with columnView[rowIndexes[b]]
	IntComparator comp2 = new IntComparator() {
		public int compare(int a, int b) {
			double av = columnView.getQuick(g[a]);
			double bv = columnView.getQuick(g[b]);
			return av<bv ? -1 : (av==bv ? 0 : 1);
		}
	};

	// compare splitter[a] with splitter[b]
	IntComparator comp3 = new IntComparator() {
		public int compare(int a, int b) {
			double av = splitters[a];
			double bv = splitters[b];
			return av<bv ? -1 : (av==bv ? 0 : 1);
		}
	};

	// generic partitioning does the main work of reordering row indexes
	org.apache.mahout.matrix.Partitioning.genericPartition(rowFrom,rowTo,splitFrom,splitTo,splitIndexes,comp,comp2,comp3,swapper);
}
/**
Same as {@link org.apache.mahout.matrix.Partitioning#partition(int[],int,int,int[],int,int,int[])}
except that it <i>synchronously</i> partitions the rows of the given matrix by the values of the given matrix column;
This is essentially the same as partitioning a list of composite objects by some instance variable;
In other words, two entire rows of the matrix are swapped, whenever two column values indicate so.
<p>
Let's say, a "row" is an "object" (tuple, d-dimensional point).
A "column" is the list of "object" values of a given variable (field, dimension).
A "matrix" is a list of "objects" (tuples, points).
<p>
Now, rows (objects, tuples) are partially sorted according to their values in one given variable (dimension).
Two entire rows of the matrix are swapped, whenever two column values indicate so.
<p>
Note that arguments are not checked for validity.
<p>
<b>Example:</b> 
<table border="1" cellspacing="0">
  <tr nowrap> 
	<td valign="top"><tt>8 x 3 matrix:<br>
	  23, 22, 21<br>
	  20, 19, 18<br>
	  17, 16, 15<br>
	  14, 13, 12<br>
	  11, 10, 9<br>
	  8,  7,  6<br>
	  5,  4,  3<br>
	  2,  1,  0 </tt></td>
	<td align="left" valign="top"> 
	    <tt>column = 0;<br>
		splitters = {5,10,12}<br>
		partition(matrix,column,splitters,splitIndexes);<br>
		==><br>
		splitIndexes == {0, 2, 3}</tt></p>
	  </td>
	<td valign="top">
	  The matrix IS NOT REORDERED.<br>
	  The new VIEW IS REORDERED:<br>
	  <tt>8 x 3 matrix:<br>
	  2,  1,  0<br>
	  5,  4,  3<br>
	  8,  7,  6<br>
	  11, 10, 9<br>
	  23, 22, 21<br>
	  20, 19, 18<br>
	  17, 16, 15<br>
	  14, 13, 12 </tt></td>
  </tr>
</table>
@param matrix the matrix to be partitioned.
@param column the index of the column to partition on.
@param splitters the values at which the rows shall be split into intervals.
	Must be sorted ascending and must not contain multiple identical values.
	These preconditions are not checked; be sure that they are met.
 
@param splitIndexes a list into which this method fills the indexes of rows delimiting intervals.
Therefore, must satisfy <tt>splitIndexes.length >= splitters.length</tt>.

@return a new matrix view having rows partitioned by the given column and splitters.
*/
public static DoubleMatrix2D partition(DoubleMatrix2D matrix, int column, final double[] splitters, int[] splitIndexes) {
	int rowFrom = 0;
	int rowTo = matrix.rows()-1;
	int splitFrom = 0;
	int splitTo = splitters.length-1;
	int[] rowIndexes = new int[matrix.rows()]; // row indexes to reorder instead of matrix itself
	for (int i=rowIndexes.length; --i >= 0; ) rowIndexes[i] = i;

	partition(matrix,rowIndexes,rowFrom,rowTo,column,splitters,splitFrom,splitTo,splitIndexes);

	// take all columns in the original order
	int[] columnIndexes = new int[matrix.columns()];
	for (int i=columnIndexes.length; --i >= 0; ) columnIndexes[i] = i;

	// view the matrix according to the reordered row indexes
	return matrix.viewSelection(rowIndexes,columnIndexes);
}
/**
Same as {@link #partition(int[],int,int,int[],int,int,int[])}
except that it <i>synchronously</i> partitions the rows of the given matrix by the values of the given matrix column;
This is essentially the same as partitioning a list of composite objects by some instance variable;
In other words, two entire rows of the matrix are swapped, whenever two column values indicate so.
<p>
Let's say, a "row" is an "object" (tuple, d-dimensional point).
A "column" is the list of "object" values of a given variable (field, dimension).
A "matrix" is a list of "objects" (tuples, points).
<p>
Now, rows (objects, tuples) are partially sorted according to their values in one given variable (dimension).
Two entire rows of the matrix are swapped, whenever two column values indicate so.
<p>
Of course, the column must not be a column of a different matrix.
More formally, there must hold: <br>
There exists an <tt>i</tt> such that <tt>matrix.viewColumn(i)==column</tt>.
<p>
Note that arguments are not checked for validity.
<p>
<b>Example:</b> 
<table border="1" cellspacing="0">
  <tr nowrap> 
	<td valign="top"><tt>8 x 3 matrix:<br>
	  23, 22, 21<br>
	  20, 19, 18<br>
	  17, 16, 15<br>
	  14, 13, 12<br>
	  11, 10, 9<br>
	  8,  7,  6<br>
	  5,  4,  3<br>
	  2,  1,  0 </tt></td>
	<td align="left"> 
	  <p><tt>column = matrix.viewColumn(0);<br>
		a = 0;<br>
		b = column.size()-1;</tt><tt><br>
		splitters={5,10,12}<br>
		c=0; <br>
		d=splitters.length-1;</tt><tt><br>
		partition(matrix,column,a,b,splitters,c,d,splitIndexes);<br>
		==><br>
		splitIndexes == {0, 2, 3}</tt></p>
	  </td>
	<td valign="top"><tt>8 x 3 matrix:<br>
	  2,  1,  0<br>
	  5,  4,  3<br>
	  8,  7,  6<br>
	  11, 10, 9<br>
	  23, 22, 21<br>
	  20, 19, 18<br>
	  17, 16, 15<br>
	  14, 13, 12 </tt></td>
  </tr>
</table>
*/
private static void xPartitionOld(DoubleMatrix2D matrix, DoubleMatrix1D column, int from, int to, double[] splitters, int splitFrom, int splitTo, int[] splitIndexes) {
	/*
	double splitter; // int, double --> template type dependent
	
	if (splitFrom>splitTo) return; // nothing to do
	if (from>to) { // all bins are empty
		from--;
		for (int i = splitFrom; i<=splitTo; ) splitIndexes[i++] = from;
		return;
	}
	
	// Choose a partition (pivot) index, m
	// Ideally, the pivot should be the median, because a median splits a list into two equal sized sublists.
	// However, computing the median is expensive, so we use an approximation.
	int medianIndex;
	if (splitFrom==splitTo) { // we don't really have a choice
		medianIndex = splitFrom;
	}
	else { // we do have a choice
		int m = (from+to) / 2;       // Small arrays, middle element
		int len = to-from+1;
		if (len > SMALL) {
		    int l = from;
		    int n = to;
		    if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
				int s = len/8;
				l = med3(column, l,     l+s, l+2*s);
				m = med3(column, m-s,   m,   m+s);
				n = med3(column, n-2*s, n-s, n);
		    }
		    m = med3(column, l, m, n); // Mid-size, pseudomedian of 3
		}
		
		// Find the splitter closest to the pivot, i.e. the splitter that best splits the list into two equal sized sublists.
		medianIndex = org.apache.mahout.matrix.Sorting.binarySearchFromTo(splitters,column.getQuick(m),splitFrom,splitTo);
		if (medianIndex < 0) medianIndex = -medianIndex - 1; // not found
		if (medianIndex > splitTo) medianIndex = splitTo; // not found, one past the end
		
	}
	splitter = splitters[medianIndex];

	// Partition the list according to the splitter, i.e.
	// Establish invariant: list[i] < splitter <= list[j] for i=from..medianIndex and j=medianIndex+1 .. to
	int	splitIndex = xPartitionOld(matrix,column,from,to,splitter);
	splitIndexes[medianIndex] = splitIndex;

	// Optimization: Handle special cases to cut down recursions.
	if (splitIndex < from) { // no element falls into this bin
		// all bins with splitters[i] <= splitter are empty
		int i = medianIndex-1;
		while (i>=splitFrom && (!(splitter < splitters[i]))) splitIndexes[i--] = splitIndex;
		splitFrom = medianIndex+1;
	}
	else if (splitIndex >= to) { // all elements fall into this bin
		// all bins with splitters[i] >= splitter are empty
		int i = medianIndex+1;
		while (i<=splitTo && (!(splitter > splitters[i]))) splitIndexes[i++] = splitIndex;
		splitTo = medianIndex-1;
	}

	// recursively partition left half
	if (splitFrom <= medianIndex-1) {
		xPartitionOld(matrix, column, from,         splitIndex, splitters, splitFrom, medianIndex-1,  splitIndexes);
	}
	
	// recursively partition right half
	if (medianIndex+1 <= splitTo) {
		xPartitionOld(matrix, column, splitIndex+1, to,         splitters, medianIndex+1,  splitTo,   splitIndexes);
	}
	*/
}
/**
 * Same as {@link #partition(int[],int,int,int)} 
 * except that it <i>synchronously</i> partitions the rows of the given matrix by the values of the given matrix column;
 * This is essentially the same as partitioning a list of composite objects by some instance variable;
 * In other words, two entire rows of the matrix are swapped, whenever two column values indicate so.
 * <p>
 * Let's say, a "row" is an "object" (tuple, d-dimensional point).
 * A "column" is the list of "object" values of a given variable (field, dimension).
 * A "matrix" is a list of "objects" (tuples, points).
 * <p>
 * Now, rows (objects, tuples) are partially sorted according to their values in one given variable (dimension).
 * Two entire rows of the matrix are swapped, whenever two column values indicate so.
 * <p>
 * Of course, the column must not be a column of a different matrix.
 * More formally, there must hold: <br>
 * There exists an <tt>i</tt> such that <tt>matrix.viewColumn(i)==column</tt>.
 *
 * Note that arguments are not checked for validity.
 */
private static int xPartitionOld(DoubleMatrix2D matrix, DoubleMatrix1D column, int from, int to, double splitter) {
	/*
	double element;  // int, double --> template type dependent
	for (int i=from-1; ++i<=to; ) {
		element = column.getQuick(i);
		if (element < splitter) {
			// swap x[i] with x[from]
			matrix.swapRows(i,from);
			from++;
		}
	}
	return from-1;
	*/
	return 0;
}
}
