/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.impl;

import org.apache.mahout.matrix.matrix.DoubleFactory2D;
import org.apache.mahout.matrix.matrix.DoubleMatrix2D;
/**
Benchmarks the performance of matrix algorithms.

@author wolfgang.hoschek@cern.ch
@version 1.0, 09/24/99
*/
class Benchmark {
/**
 * Makes this class non instantiable, but still let's others inherit from it.
 */
protected Benchmark() {
	throw new RuntimeException("Non instantiable");
}
/**
 * Runs a bench on matrices holding double elements.
 */
public static void benchmark(int runs, int size, String kind, boolean print, int initialCapacity, double minLoadFactor, double maxLoadFactor, double percentNonZero) {
	// certain loops need to be constructed so that the jitter can't optimize them away and we get fantastic numbers.
	// this involves primarly read-loops

	org.apache.mahout.matrix.Timer timer1 = new org.apache.mahout.matrix.Timer();
	org.apache.mahout.matrix.Timer timer2 = new org.apache.mahout.matrix.Timer();
	org.apache.mahout.matrix.Timer timer3 = new org.apache.mahout.matrix.Timer();
	org.apache.mahout.matrix.Timer timer4 = new org.apache.mahout.matrix.Timer();
	org.apache.mahout.matrix.Timer timer5 = new org.apache.mahout.matrix.Timer();
	org.apache.mahout.matrix.Timer timer6 = new org.apache.mahout.matrix.Timer();

	DoubleMatrix2D  matrix = null;
	if (kind.equals("sparse")) matrix = new SparseDoubleMatrix2D(size,size,initialCapacity,minLoadFactor,maxLoadFactor);
	else if (kind.equals("dense")) matrix = org.apache.mahout.matrix.matrix.DoubleFactory2D.dense.make(size,size);
	//else if (kind.equals("denseArray")) matrix = new DoubleArrayMatrix2D(size,size);
	else throw new RuntimeException("unknown kind");
	
	System.out.println("\nNow initializing...");
	//Matrix AJ = new Matrix(columnwise,3);
	//Basic.random(matrix, new org.apache.mahout.jet.random.Uniform(new org.apache.mahout.jet.random.engine.MersenneTwister()));
	double value = 2;
	DoubleMatrix2D tmp = DoubleFactory2D.dense.sample(matrix.rows(), matrix.columns(), value, percentNonZero);
	matrix.assign(tmp);
	tmp = null;
	/*
	long NN = matrix.size();
	int nn = (int) (NN*percentNonZero);
	long[] nonZeroIndexes = new long[nn];
	org.apache.mahout.jet.random.sampling.RandomSampler sampler = new org.apache.mahout.jet.random.sampling.RandomSampler(nn,NN,0,new org.apache.mahout.jet.random.engine.MersenneTwister());
	sampler.nextBlock(nn,nonZeroIndexes,0);
	for (int i=nn; --i >=0; ) {
		int row = (int) (nonZeroIndexes[i]/size);
		int column = (int) (nonZeroIndexes[i]%size);
		matrix.set(row,column, value);
	}
	*/

	/*
	timer1.start();
	for (int i=0; i<runs; i++) {
		LUDecomposition LU = new LUDecomposition(matrix);
	}
	timer1.stop();
	timer1.display();

	{
		Jama.Matrix jmatrix = new Jama.Matrix(matrix.toArray());
		timer2.start();
		for (int i=0; i<runs; i++) {
			Jama.LUDecomposition LU = new Jama.LUDecomposition(jmatrix);
		}
		timer2.stop();
		timer2.display();
	}
	*/
	System.out.println("\ntesting...");
	if (print) System.out.println(matrix);
	DoubleMatrix2D dense = DoubleFactory2D.dense.make(size,size);
	dense.assign(matrix);
	if (! dense.equals(matrix)) throw new InternalError();
	DoubleMatrix2D ADense = dense.copy();
	DoubleMatrix2D BDense = dense.copy();
	DoubleMatrix2D CDense = dense.copy();
	ADense.zMult(BDense,CDense);
	System.out.println("\nNext testing...");
	/*
	{
		timer6.start();
		double a = cubicLoop(runs,size);
		timer6.stop();
		timer6.display();
		System.out.println(a);
	}
	*/
	

	{
		DoubleMatrix2D A = matrix.copy();
		DoubleMatrix2D B = matrix.copy();
		//DoubleMatrix2D C = Basic.product(A,B);
		DoubleMatrix2D C = matrix.copy();
		A.zMult(B,C);
		if (! (C.equals(CDense))) throw new InternalError();
		C.assign(matrix);
		System.out.println("\nNow benchmarking...");
		
		timer3.start();
		for (int i=0; i<runs; i++) {
			A.zMult(B,C);
		}
		timer3.stop();
		timer3.display();
		int m = A.rows();
		int n = A.columns();
		int p = B.rows();
		int reps = runs;
		double mflops = 1.0e-3*(2.0*m*n*p*reps)/timer3.millis();
		System.out.println("mflops: "+mflops);
	}
	
	/*
	{
		DoubleMatrix2D A = matrix.like().assign(value);
		DoubleMatrix2D B = matrix.like().assign(value);
		DoubleMatrix2D C = Basic.product(A,B);
		timer5.start();
		for (int i=0; i<runs; i++) {
			org.apache.mahout.matrix.matrix.Blas.matrixMultiply(A,B,C);
		}
		timer5.stop();
		timer5.display();
	}
	*/
	

/*
{
		Jama.Matrix A = new Jama.Matrix(size,size);
		Jama.Matrix B = new Jama.Matrix(size,size);
		Jama.Matrix C;
		timer4.start();
		for (int i=0; i<runs; i++) {
			C = A.times(B);
		}
		timer4.stop();
		timer4.display();
	}
*/

	if (print) System.out.println(matrix);

	System.out.println("bye bye.");
}
/**
 * 
 */
protected static double cubicLoop(int runs, int size) {
	double a = 1.123;
	double b = 1.000000000012345;
	for (int r=0; r<runs; r++) {
		for (int i=size; --i >= 0; ) {
			for (int j=size; --j >= 0; ) {
				for (int k=size; --k >= 0; ) {
					a *= b;
				}
			}
		}
	}
	return a;
}
/**
 * Benchmarks various matrix methods.
 */
public static void main(String args[]) {
	int runs = Integer.parseInt(args[0]);
	int rows = Integer.parseInt(args[1]);
	int columns = Integer.parseInt(args[2]);
	//int size = Integer.parseInt(args[3]);
	//boolean isSparse = args[4].equals("sparse");
	String kind = args[3];
	int initialCapacity = Integer.parseInt(args[4]);
	double minLoadFactor = new Double(args[5]).doubleValue();
	double maxLoadFactor = new Double(args[6]).doubleValue();
	boolean print = args[7].equals("print");
	double initialValue = new Double(args[8]).doubleValue();
	int size = rows;
	
	benchmark(runs,size,kind,print,initialCapacity,minLoadFactor,maxLoadFactor,initialValue);
}
}
