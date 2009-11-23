/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.impl;

import org.apache.mahout.matrix.matrix.ObjectMatrix1D;
import org.apache.mahout.matrix.matrix.ObjectMatrix2D;
/**
Dense 1-d matrix (aka <i>vector</i>) holding <tt>Object</tt> elements.
First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
<p>
<b>Implementation:</b>
<p>
Internally holds one single contigous one-dimensional array. 
Note that this implementation is not synchronized.
<p>
<b>Memory requirements:</b>
<p>
<tt>memory [bytes] = 8*size()</tt>.
Thus, a 1000000 matrix uses 8 MB.
<p>
<b>Time complexity:</b>
<p>
<tt>O(1)</tt> (i.e. constant time) for the basic operations
<tt>get</tt>, <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
<p>
@author wolfgang.hoschek@cern.ch
@version 1.0, 09/24/99
*/
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class DenseObjectMatrix1D extends ObjectMatrix1D {
	/**
	  * The elements of this matrix.
	  */
	protected Object[] elements;
/**
 * Constructs a matrix with a copy of the given values.
 * The values are copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
 *
 * @param values The values to be filled into the new matrix.
 */
public DenseObjectMatrix1D(Object[] values) {
	this(values.length);
	assign(values);
}
/**
 * Constructs a matrix with a given number of cells.
 * All entries are initially <tt>0</tt>.
 * @param size the number of cells the matrix shall have.
 * @throws IllegalArgumentException if <tt>size<0</tt>.
 */
public DenseObjectMatrix1D(int size) {
	setUp(size);
	this.elements = new Object[size];
}
/**
 * Constructs a matrix view with the given parameters.
 * @param size the number of cells the matrix shall have.
 * @param elements the cells.
 * @param zero the index of the first element.
 * @param stride the number of indexes between any two elements, i.e. <tt>index(i+1)-index(i)</tt>.
 * @throws IllegalArgumentException if <tt>size<0</tt>.
 */
protected DenseObjectMatrix1D(int size, Object[] elements, int zero, int stride) {
	setUp(size,zero,stride);
	this.elements = elements;
	this.isNoView = false;
}
/**
 * Sets all cells to the state specified by <tt>values</tt>.
 * <tt>values</tt> is required to have the same number of cells as the receiver.
 * <p>
 * The values are copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
 *
 * @param    values the values to be filled into the cells.
 * @return <tt>this</tt> (for convenience only).
 * @throws IllegalArgumentException if <tt>values.length != size()</tt>.
 */
public ObjectMatrix1D assign(Object[] values) {
	if (isNoView) {
		if (values.length != size) throw new IllegalArgumentException("Must have same number of cells: length="+values.length+"size()="+size());
		System.arraycopy(values, 0, this.elements, 0, values.length);
	}
	else {
		super.assign(values);
	}
	return this;
}
/**
Assigns the result of a function to each cell; <tt>x[i] = function(x[i])</tt>.
(Iterates downwards from <tt>[size()-1]</tt> to <tt>[0]</tt>).
<p>
<b>Example:</b>
<pre>
// change each cell to its sine
matrix =   0.5      1.5      2.5       3.5 
matrix.assign(org.apache.mahout.jet.math.Functions.sin);
-->
matrix ==  0.479426 0.997495 0.598472 -0.350783
</pre>
For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.

@param function a function object taking as argument the current cell's value.
@return <tt>this</tt> (for convenience only).
@see org.apache.mahout.jet.math.Functions
*/
public ObjectMatrix1D assign(org.apache.mahout.matrix.function.ObjectFunction function) {
	int s=stride;
	int i=index(0);
	Object[] elems = this.elements;
	if (elements==null) throw new InternalError();

	// the general case x[i] = f(x[i])
	for (int k=size; --k >= 0; ) {
		elems[i] = function.apply(elems[i]);
		i += s;
	}
	return this;
}
/**
 * Replaces all cell values of the receiver with the values of another matrix.
 * Both matrices must have the same size.
 * If both matrices share the same cells (as is the case if they are views derived from the same matrix) and intersect in an ambiguous way, then replaces <i>as if</i> using an intermediate auxiliary deep copy of <tt>other</tt>.
 *
 * @param     source   the source matrix to copy from (may be identical to the receiver).
 * @return <tt>this</tt> (for convenience only).
 * @throws	IllegalArgumentException if <tt>size() != other.size()</tt>.
 */
public ObjectMatrix1D assign(ObjectMatrix1D source) {
	// overriden for performance only
	if (! (source instanceof DenseObjectMatrix1D)) {
		return super.assign(source);
	}
	DenseObjectMatrix1D other = (DenseObjectMatrix1D) source;
	if (other==this) return this;
	checkSize(other);
	if (isNoView && other.isNoView) { // quickest
		System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
		return this;
	}
	if (haveSharedCells(other)) {
		ObjectMatrix1D c = other.copy();
		if (! (c instanceof DenseObjectMatrix1D)) { // should not happen
			return super.assign(source);
		}
		other = (DenseObjectMatrix1D) c;
	}

	final Object[] elems = this.elements;
	final Object[] otherElems = other.elements;
	if (elements==null || otherElems==null) throw new InternalError();
	int s = this.stride;
	int ys = other.stride;

	int index = index(0);
	int otherIndex = other.index(0);
	for (int k=size; --k >= 0; ) {
		elems[index] = otherElems[otherIndex];
		index += s;
		otherIndex += ys;
	}
	return this;
}
/**
Assigns the result of a function to each cell; <tt>x[i] = function(x[i],y[i])</tt>.
(Iterates downwards from <tt>[size()-1]</tt> to <tt>[0]</tt>).
<p>
<b>Example:</b>
<pre>
// assign x[i] = x[i]<sup>y[i]</sup>
m1 = 0 1 2 3;
m2 = 0 2 4 6;
m1.assign(m2, org.apache.mahout.jet.math.Functions.pow);
-->
m1 == 1 1 16 729

// for non-standard functions there is no shortcut: 
m1.assign(m2,
&nbsp;&nbsp;&nbsp;new ObjectObjectFunction() {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;public Object apply(Object x, Object y) { return Math.pow(x,y); }
&nbsp;&nbsp;&nbsp;}
);
</pre>
For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.

@param y the secondary matrix to operate on.
@param function a function object taking as first argument the current cell's value of <tt>this</tt>,
and as second argument the current cell's value of <tt>y</tt>,
@return <tt>this</tt> (for convenience only).
@throws	IllegalArgumentException if <tt>size() != y.size()</tt>.
@see org.apache.mahout.jet.math.Functions
*/
public ObjectMatrix1D assign(ObjectMatrix1D y, org.apache.mahout.matrix.function.ObjectObjectFunction function) {
	// overriden for performance only
	if (! (y instanceof DenseObjectMatrix1D)) {
		return super.assign(y,function);
	}
	DenseObjectMatrix1D other = (DenseObjectMatrix1D) y;
	checkSize(y);
	final Object[] elems = this.elements;
	final Object[] otherElems = other.elements;
	if (elements==null || otherElems==null) throw new InternalError();
	int s = this.stride;
	int ys = other.stride;

	int index = index(0);
	int otherIndex = other.index(0);

	// the general case x[i] = f(x[i],y[i])		
	for (int k=size; --k >= 0; ) {
		elems[index] = function.apply(elems[index], otherElems[otherIndex]);
		index += s;
		otherIndex += ys;
	}
	return this;
}
/**
 * Returns the matrix cell value at coordinate <tt>index</tt>.
 *
 * <p>Provided with invalid parameters this method may return invalid objects without throwing any exception.
 * <b>You should only use this method when you are absolutely sure that the coordinate is within bounds.</b>
 * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
 *
 * @param     index   the index of the cell.
 * @return    the value of the specified cell.
 */
public Object getQuick(int index) {
	//if (debug) if (index<0 || index>=size) checkIndex(index);
	//return elements[index(index)];
	// manually inlined:
	return elements[zero + index*stride];
}
/**
 * Returns <tt>true</tt> if both matrices share at least one identical cell.
 */
protected boolean haveSharedCellsRaw(ObjectMatrix1D other) {
	if (other instanceof SelectedDenseObjectMatrix1D) {
		SelectedDenseObjectMatrix1D otherMatrix = (SelectedDenseObjectMatrix1D) other;
		return this.elements==otherMatrix.elements;
	}
	else if (other instanceof DenseObjectMatrix1D) {
		DenseObjectMatrix1D otherMatrix = (DenseObjectMatrix1D) other;
		return this.elements==otherMatrix.elements;
	}
	return false;
}
/**
 * Returns the position of the element with the given relative rank within the (virtual or non-virtual) internal 1-dimensional array.
 * You may want to override this method for performance.
 *
 * @param     rank   the rank of the element.
 */
protected int index(int rank) {
	// overriden for manual inlining only
	//return _offset(_rank(rank));
	return zero + rank*stride;
}
/**
 * Construct and returns a new empty matrix <i>of the same dynamic type</i> as the receiver, having the specified size.
 * For example, if the receiver is an instance of type <tt>DenseObjectMatrix1D</tt> the new matrix must also be of type <tt>DenseObjectMatrix1D</tt>,
 * if the receiver is an instance of type <tt>SparseObjectMatrix1D</tt> the new matrix must also be of type <tt>SparseObjectMatrix1D</tt>, etc.
 * In general, the new matrix should have internal parametrization as similar as possible.
 *
 * @param size the number of cell the matrix shall have.
 * @return  a new empty matrix of the same dynamic type.
 */
public ObjectMatrix1D like(int size) {
	return new DenseObjectMatrix1D(size);
}
/**
 * Construct and returns a new 2-d matrix <i>of the corresponding dynamic type</i>, entirelly independent of the receiver.
 * For example, if the receiver is an instance of type <tt>DenseObjectMatrix1D</tt> the new matrix must be of type <tt>DenseObjectMatrix2D</tt>,
 * if the receiver is an instance of type <tt>SparseObjectMatrix1D</tt> the new matrix must be of type <tt>SparseObjectMatrix2D</tt>, etc.
 *
 * @param rows the number of rows the matrix shall have.
 * @param columns the number of columns the matrix shall have.
 * @return  a new matrix of the corresponding dynamic type.
 */
public ObjectMatrix2D like2D(int rows, int columns) {
	return new DenseObjectMatrix2D(rows,columns);
}
/**
 * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
 *
 * <p>Provided with invalid parameters this method may access illegal indexes without throwing any exception.
 * <b>You should only use this method when you are absolutely sure that the coordinate is within bounds.</b>
 * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
 *
 * @param     index   the index of the cell.
 * @param    value the value to be filled into the specified cell.
 */
public void setQuick(int index, Object value) {
	//if (debug) if (index<0 || index>=size) checkIndex(index);
	//elements[index(index)] = value;
	// manually inlined:
	elements[zero + index*stride] = value;
}
/**
Swaps each element <tt>this[i]</tt> with <tt>other[i]</tt>.
@throws IllegalArgumentException if <tt>size() != other.size()</tt>.
*/
public void swap(ObjectMatrix1D other) {
	// overriden for performance only
	if (! (other instanceof DenseObjectMatrix1D)) {
		super.swap(other);
	}
	DenseObjectMatrix1D y = (DenseObjectMatrix1D) other;
	if (y==this) return;
	checkSize(y);
	
	final Object[] elems = this.elements;
	final Object[] otherElems = y.elements;
	if (elements==null || otherElems==null) throw new InternalError();
	int s = this.stride;
	int ys = y.stride;

	int index = index(0);
	int otherIndex = y.index(0);
	for (int k=size; --k >= 0; ) {
		Object tmp = elems[index];
		elems[index] = otherElems[otherIndex];
		otherElems[otherIndex] = tmp;
		index += s;
		otherIndex += ys;
	}
	return;
}
/**
Fills the cell values into the specified 1-dimensional array.
The values are copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
After this call returns the array <tt>values</tt> has the form 
<br>
<tt>for (int i=0; i < size(); i++) values[i] = get(i);</tt>

@throws IllegalArgumentException if <tt>values.length < size()</tt>.
*/
public void toArray(Object[] values) {
	if (values.length < size) throw new IllegalArgumentException("values too small");
	if (this.isNoView) System.arraycopy(this.elements,0,values,0,this.elements.length);
	else super.toArray(values);
}
/**
 * Construct and returns a new selection view.
 *
 * @param offsets the offsets of the visible elements.
 * @return  a new view.
 */
protected ObjectMatrix1D viewSelectionLike(int[] offsets) {
	return new SelectedDenseObjectMatrix1D(this.elements,offsets);
}
}
