/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix;

import org.apache.mahout.matrix.matrix.impl.DenseObjectMatrix3D;
import org.apache.mahout.matrix.matrix.impl.SparseObjectMatrix3D;
/**
Factory for convenient construction of 3-d matrices holding <tt>Object</tt> cells. 
Use idioms like <tt>ObjectFactory3D.dense.make(4,4,4)</tt> to construct dense matrices, 
<tt>ObjectFactory3D.sparse.make(4,4,4)</tt> to construct sparse matrices.

If the factory is used frequently it might be useful to streamline the notation. 
For example by aliasing:
<table>
<td class="PRE"> 
<pre>
ObjectFactory3D F = ObjectFactory3D.dense;
F.make(4,4,4);
...
</pre>
</td>
</table>

@author wolfgang.hoschek@cern.ch
@version 1.0, 09/24/99
*/
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class ObjectFactory3D extends org.apache.mahout.matrix.PersistentObject {
	/**
	 * A factory producing dense matrices.
	 */
	public static final ObjectFactory3D dense  = new ObjectFactory3D();

	/**
	 * A factory producing sparse matrices.
	 */
	public static final ObjectFactory3D sparse = new ObjectFactory3D();
/**
 * Makes this class non instantiable, but still let's others inherit from it.
 */
protected ObjectFactory3D() {}
/**
 * Constructs a matrix with the given cell values.
 * <tt>values</tt> is required to have the form <tt>values[slice][row][column]</tt>
 * and have exactly the same number of slices, rows and columns as the receiver.
 * <p>
 * The values are copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
 *
 * @param    values the values to be filled into the cells.
 * @return <tt>this</tt> (for convenience only).
 * @throws IllegalArgumentException if <tt>values.length != slices() || for any 0 &lt;= slice &lt; slices(): values[slice].length != rows()</tt>.
 * @throws IllegalArgumentException if <tt>for any 0 &lt;= column &lt; columns(): values[slice][row].length != columns()</tt>.
 */
public ObjectMatrix3D make(Object[][][] values) {
	if (this==sparse) return new SparseObjectMatrix3D(values);
	return new DenseObjectMatrix3D(values);
}
/**
 * Constructs a matrix with the given shape, each cell initialized with zero.
 */
public ObjectMatrix3D make(int slices, int rows, int columns) {
	if (this==sparse) return new SparseObjectMatrix3D(slices,rows,columns);
	return new DenseObjectMatrix3D(slices,rows,columns);
}
/**
 * Constructs a matrix with the given shape, each cell initialized with the given value.
 */
public ObjectMatrix3D make(int slices, int rows, int columns, Object initialValue) {
	return make(slices,rows,columns).assign(initialValue);
}
}
