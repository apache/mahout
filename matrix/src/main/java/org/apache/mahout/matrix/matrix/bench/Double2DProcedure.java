/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.colt.matrix.bench;

import org.apache.mahout.colt.matrix.DoubleMatrix2D;

abstract class Double2DProcedure implements TimerProcedure {
	public DoubleMatrix2D A;
	public DoubleMatrix2D B;
	public DoubleMatrix2D C;
	public DoubleMatrix2D D;
/**
 * The number of operations a single call to "apply" involves.
 */
public double operations() {
	return A.rows()*A.columns() / 1.0E6;
}
/**
 * Sets the matrices to operate upon.
 */
public void setParameters(DoubleMatrix2D A, DoubleMatrix2D B) {
	this.A=A;
	this.B=B;
	this.C=A.copy();
	this.D=B.copy();
}
}
