/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.colt.matrix.impl;

import org.apache.mahout.colt.map.AbstractIntObjectMap;
import org.apache.mahout.colt.map.OpenIntObjectHashMap;
import org.apache.mahout.colt.matrix.ObjectMatrix2D;
import org.apache.mahout.colt.matrix.ObjectMatrix3D;
/**
Sparse hashed 3-d matrix holding <tt>Object</tt> elements.
First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
<p>
<b>Implementation:</b>
<p>
Note that this implementation is not synchronized.
Uses a {@link org.apache.mahout.colt.map.OpenIntObjectHashMap}, which is a compact and performant hashing technique.
<p>
<b>Memory requirements:</b>
<p>
Cells that
<ul>
<li>are never set to non-zero values do not use any memory.
<li>switch from zero to non-zero state do use memory.
<li>switch back from non-zero to zero state also do use memory. However, their memory is automatically reclaimed from time to time. It can also manually be reclaimed by calling {@link #trimToSize()}.
</ul>
<p>
worst case: <tt>memory [bytes] = (1/minLoadFactor) * nonZeros * 13</tt>.
<br>best  case: <tt>memory [bytes] = (1/maxLoadFactor) * nonZeros * 13</tt>.
<br>Where <tt>nonZeros = cardinality()</tt> is the number of non-zero cells.
Thus, a 100 x 100 x 100 matrix with minLoadFactor=0.25 and maxLoadFactor=0.5 and 1000000 non-zero cells consumes between 25 MB and 50 MB.
The same 100 x 100 x 100 matrix with 1000 non-zero cells consumes between 25 and 50 KB.
<p>
<b>Time complexity:</b>
<p>
This class offers <i>expected</i> time complexity <tt>O(1)</tt> (i.e. constant time) for the basic operations
<tt>get</tt>, <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>
assuming the hash function disperses the elements properly among the buckets.
Otherwise, pathological cases, although highly improbable, can occur, degrading performance to <tt>O(N)</tt> in the worst case.
As such this sparse class is expected to have no worse time complexity than its dense counterpart {@link DenseObjectMatrix2D}.
However, constant factors are considerably larger.
<p>
Cells are internally addressed in (in decreasing order of significance): slice major, row major, column major.
Applications demanding utmost speed can exploit this fact.
Setting/getting values in a loop slice-by-slice, row-by-row, column-by-column is quicker than, for example, column-by-column, row-by-row, slice-by-slice.
Thus
<pre>
   for (int slice=0; slice < slices; slice++) {
	  for (int row=0; row < rows; row++) {
	     for (int column=0; column < columns; column++) {
			matrix.setQuick(slice,row,column,someValue);
		 }		    
	  }
   }
</pre>
is quicker than
<pre>
   for (int column=0; column < columns; column++) {
	  for (int row=0; row < rows; row++) {
	     for (int slice=0; slice < slices; slice++) {
			matrix.setQuick(slice,row,column,someValue);
		 }
	  }
   }
</pre>

@see org.apache.mahout.colt.map
@see org.apache.mahout.colt.map.OpenIntObjectHashMap
@author wolfgang.hoschek@cern.ch
@version 1.0, 09/24/99
*/
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class SparseObjectMatrix3D extends ObjectMatrix3D {
	/*
	 * The elements of the matrix.
	 */
	protected AbstractIntObjectMap elements; 
/**
 * Constructs a matrix with a copy of the given values.
 * <tt>values</tt> is required to have the form <tt>values[slice][row][column]</tt>
 * and have exactly the same number of rows in in every slice and exactly the same number of columns in in every row.
 * <p>
 * The values are copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
 *
 * @param values The values to be filled into the new matrix.
 * @throws IllegalArgumentException if <tt>for any 1 &lt;= slice &lt; values.length: values[slice].length != values[slice-1].length</tt>.
 * @throws IllegalArgumentException if <tt>for any 1 &lt;= row &lt; values[0].length: values[slice][row].length != values[slice][row-1].length</tt>.
 */
public SparseObjectMatrix3D(Object[][][] values) {
	this(values.length, (values.length==0 ? 0: values[0].length), (values.length==0 ? 0: values[0].length==0 ? 0 : values[0][0].length));
	assign(values);
}
/**
 * Constructs a matrix with a given number of slices, rows and columns and default memory usage.
 * All entries are initially <tt>null</tt>.
 * @param slices the number of slices the matrix shall have.
 * @param rows the number of rows the matrix shall have.
 * @param columns the number of columns the matrix shall have.
 * @throws	IllegalArgumentException if <tt>(double)slices*columns*rows > Integer.MAX_VALUE</tt>.
 * @throws	IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
 */
public SparseObjectMatrix3D(int slices, int rows, int columns) {
	this(slices,rows,columns,slices*rows*(columns/1000),0.2,0.5);
}
/**
 * Constructs a matrix with a given number of slices, rows and columns using memory as specified.
 * All entries are initially <tt>null</tt>.
 * For details related to memory usage see {@link org.apache.mahout.colt.map.OpenIntObjectHashMap}.
 * 
 * @param slices the number of slices the matrix shall have.
 * @param rows the number of rows the matrix shall have.
 * @param columns the number of columns the matrix shall have.
 * @param initialCapacity   the initial capacity of the hash map.
 *                          If not known, set <tt>initialCapacity=0</tt> or small.     
 * @param minLoadFactor        the minimum load factor of the hash map.
 * @param maxLoadFactor        the maximum load factor of the hash map.
 * @throws	IllegalArgumentException if <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) || (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >= maxLoadFactor)</tt>.
 * @throws	IllegalArgumentException if <tt>(double)slices*columns*rows > Integer.MAX_VALUE</tt>.
 * @throws	IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
 */
public SparseObjectMatrix3D(int slices, int rows, int columns, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
	setUp(slices,rows,columns);
	this.elements = new OpenIntObjectHashMap(initialCapacity, minLoadFactor, maxLoadFactor);
}
/**
 * Constructs a view with the given parameters.
 * @param slices the number of slices the matrix shall have.
 * @param rows the number of rows the matrix shall have.
 * @param columns the number of columns the matrix shall have.
 * @param elements the cells.
 * @param sliceZero the position of the first element.
 * @param rowZero the position of the first element.
 * @param columnZero the position of the first element.
 * @param sliceStride the number of elements between two slices, i.e. <tt>index(k+1,i,j)-index(k,i,j)</tt>.
 * @param rowStride the number of elements between two rows, i.e. <tt>index(k,i+1,j)-index(k,i,j)</tt>.
 * @param columnnStride the number of elements between two columns, i.e. <tt>index(k,i,j+1)-index(k,i,j)</tt>.
 * @throws	IllegalArgumentException if <tt>(Object)slices*columns*rows > Integer.MAX_VALUE</tt>.
 * @throws	IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
 */
protected SparseObjectMatrix3D(int slices, int rows, int columns, AbstractIntObjectMap elements, int sliceZero, int rowZero, int columnZero, int sliceStride, int rowStride, int columnStride) {
	setUp(slices,rows,columns,sliceZero,rowZero,columnZero,sliceStride,rowStride,columnStride);
	this.elements = elements;
	this.isNoView = false;
}
/**
 * Returns the number of cells having non-zero values.
 */
public int cardinality() {
	if (this.isNoView) return this.elements.size();
	else return super.cardinality();
}
/**
 * Ensures that the receiver can hold at least the specified number of non-zero cells without needing to allocate new internal memory.
 * If necessary, allocates new internal memory and increases the capacity of the receiver.
 * <p>
 * This method never need be called; it is for performance tuning only.
 * Calling this method before tt>set()</tt>ing a large number of non-zero values boosts performance,
 * because the receiver will grow only once instead of potentially many times and hash collisions get less probable.
 *
 * @param   minNonZeros   the desired minimum number of non-zero cells.
 */
public void ensureCapacity(int minCapacity) {
	this.elements.ensureCapacity(minCapacity);
}
/**
 * Returns the matrix cell value at coordinate <tt>[slice,row,column]</tt>.
 *
 * <p>Provided with invalid parameters this method may return invalid objects without throwing any exception.
 * <b>You should only use this method when you are absolutely sure that the coordinate is within bounds.</b>
 * Precondition (unchecked): <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 || column&gt;=column()</tt>.
 *
 * @param     slice   the index of the slice-coordinate.
 * @param     row   the index of the row-coordinate.
 * @param     column   the index of the column-coordinate.
 * @return    the value at the specified coordinate.
 */
public Object getQuick(int slice, int row, int column) {
	//if (debug) if (slice<0 || slice>=slices || row<0 || row>=rows || column<0 || column>=columns) throw new IndexOutOfBoundsException("slice:"+slice+", row:"+row+", column:"+column);
	//return elements.get(index(slice,row,column));
	//manually inlined:
	return elements.get(sliceZero + slice*sliceStride + rowZero + row*rowStride + columnZero + column*columnStride);
}
/**
 * Returns <tt>true</tt> if both matrices share at least one identical cell.
 */
protected boolean haveSharedCellsRaw(ObjectMatrix3D other) {
	if (other instanceof SelectedSparseObjectMatrix3D) {
		SelectedSparseObjectMatrix3D otherMatrix = (SelectedSparseObjectMatrix3D) other;
		return this.elements==otherMatrix.elements;
	}
	else if (other instanceof SparseObjectMatrix3D) {
		SparseObjectMatrix3D otherMatrix = (SparseObjectMatrix3D) other;
		return this.elements==otherMatrix.elements;
	}
	return false;
}
/**
 * Returns the position of the given coordinate within the (virtual or non-virtual) internal 1-dimensional array. 
 *
 * @param     slice   the index of the slice-coordinate.
 * @param     row   the index of the row-coordinate.
 * @param     column   the index of the third-coordinate.
 */
protected int index(int slice, int row, int column) {
	//return _sliceOffset(_sliceRank(slice)) + _rowOffset(_rowRank(row)) + _columnOffset(_columnRank(column));
	//manually inlined:
	return sliceZero + slice*sliceStride + rowZero + row*rowStride + columnZero + column*columnStride;	
}
/**
 * Construct and returns a new empty matrix <i>of the same dynamic type</i> as the receiver, having the specified number of slices, rows and columns.
 * For example, if the receiver is an instance of type <tt>DenseObjectMatrix3D</tt> the new matrix must also be of type <tt>DenseObjectMatrix3D</tt>,
 * if the receiver is an instance of type <tt>SparseObjectMatrix3D</tt> the new matrix must also be of type <tt>SparseObjectMatrix3D</tt>, etc.
 * In general, the new matrix should have internal parametrization as similar as possible.
 *
 * @param slices the number of slices the matrix shall have.
 * @param rows the number of rows the matrix shall have.
 * @param columns the number of columns the matrix shall have.
 * @return  a new empty matrix of the same dynamic type.
 */
public ObjectMatrix3D like(int slices, int rows, int columns) {
	return new SparseObjectMatrix3D(slices,rows,columns); 
}
/**
 * Construct and returns a new 2-d matrix <i>of the corresponding dynamic type</i>, sharing the same cells.
 * For example, if the receiver is an instance of type <tt>DenseObjectMatrix3D</tt> the new matrix must also be of type <tt>DenseObjectMatrix2D</tt>,
 * if the receiver is an instance of type <tt>SparseObjectMatrix3D</tt> the new matrix must also be of type <tt>SparseObjectMatrix2D</tt>, etc.
 *
 * @param rows the number of rows the matrix shall have.
 * @param columns the number of columns the matrix shall have.
 * @param rowZero the position of the first element.
 * @param columnZero the position of the first element.
 * @param rowStride the number of elements between two rows, i.e. <tt>index(i+1,j)-index(i,j)</tt>.
 * @param columnStride the number of elements between two columns, i.e. <tt>index(i,j+1)-index(i,j)</tt>.
 * @return  a new matrix of the corresponding dynamic type.
 */
protected ObjectMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
	return new SparseObjectMatrix2D(rows,columns,this.elements,rowZero,columnZero,rowStride,columnStride);
}
/**
 * Sets the matrix cell at coordinate <tt>[slice,row,column]</tt> to the specified value.
 *
 * <p>Provided with invalid parameters this method may access illegal indexes without throwing any exception.
 * <b>You should only use this method when you are absolutely sure that the coordinate is within bounds.</b>
 * Precondition (unchecked): <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 || column&gt;=column()</tt>.
 *
 * @param     slice   the index of the slice-coordinate.
 * @param     row   the index of the row-coordinate.
 * @param     column   the index of the column-coordinate.
 * @param    value the value to be filled into the specified cell.
 */
public void setQuick(int slice, int row, int column, Object value) {
	//if (debug) if (slice<0 || slice>=slices || row<0 || row>=rows || column<0 || column>=columns) throw new IndexOutOfBoundsException("slice:"+slice+", row:"+row+", column:"+column);
	//int index =	index(slice,row,column);
	//manually inlined:
	int index = sliceZero + slice*sliceStride + rowZero + row*rowStride + columnZero + column*columnStride;
	if (value == null)
		this.elements.removeKey(index);
	else 
		this.elements.put(index, value);
}
/**
 * Releases any superfluous memory created by explicitly putting zero values into cells formerly having non-zero values; 
 * An application can use this operation to minimize the
 * storage of the receiver.
 * <p>
 * <b>Background:</b>
 * <p>
 * Cells that <ul>
 * <li>are never set to non-zero values do not use any memory.
 * <li>switch from zero to non-zero state do use memory.
 * <li>switch back from non-zero to zero state also do use memory. However, their memory can be reclaimed by calling <tt>trimToSize()</tt>.
 * </ul>
 * A sequence like <tt>set(s,r,c,5); set(s,r,c,0);</tt>
 * sets a cell to non-zero state and later back to zero state.
 * Such as sequence generates obsolete memory that is automatically reclaimed from time to time or can manually be reclaimed by calling <tt>trimToSize()</tt>.
 * Putting zeros into cells already containing zeros does not generate obsolete memory since no memory was allocated to them in the first place.
 */
public void trimToSize() {
	this.elements.trimToSize();
}
/**
 * Construct and returns a new selection view.
 *
 * @param sliceOffsets the offsets of the visible elements.
 * @param rowOffsets the offsets of the visible elements.
 * @param columnOffsets the offsets of the visible elements.
 * @return  a new view.
 */
protected ObjectMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
	return new SelectedSparseObjectMatrix3D(this.elements,sliceOffsets,rowOffsets,columnOffsets,0);
}
}
