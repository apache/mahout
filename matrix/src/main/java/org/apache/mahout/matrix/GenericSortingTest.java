/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix;

import org.apache.mahout.matrix.function.IntComparator;
/**
Demonstrates how to use {@link Sort}.

@author wolfgang.hoschek@cern.ch
@version 1.0, 03-Jul-99
*/
class GenericSortingTest extends Object {
/**
 * Makes this class non instantiable, but still let's others inherit from it.
 */
protected GenericSortingTest() {}
/**
 * Just a demo.
 */
public static void demo1() {
	final int[] x;
	final double[] y;
	final double[] z;

	x = new int[]    {3,   2,   1  };
	y = new double[] {3.0, 2.0, 1.0};
	z = new double[] {6.0, 7.0, 8.0};

	Swapper swapper = new Swapper() {
		public void swap(int a, int b) {
			int t1;	double t2, t3;
			t1 = x[a]; x[a] = x[b];	x[b] = t1;
			t2 = y[a]; y[a] = y[b]; y[b] = t2;
			t3 = z[a]; z[a] = z[b];	z[b] = t3;
		}
	}; 

	IntComparator comp = new IntComparator() {
		public int compare(int a, int b) {
			return x[a]==x[b] ? 0 : (x[a]<x[b] ? -1 : 1);
		}
	};

	System.out.println("before:");
	System.out.println("X="+Arrays.toString(x));
	System.out.println("Y="+Arrays.toString(y));
	System.out.println("Z="+Arrays.toString(z));

			
	int from = 0;
	int to = x.length;
	GenericSorting.quickSort(from, to, comp, swapper);

	System.out.println("after:");
	System.out.println("X="+Arrays.toString(x));
	System.out.println("Y="+Arrays.toString(y));
	System.out.println("Z="+Arrays.toString(z));
	System.out.println("\n\n");
}
/**
 * Just a demo.
 */
public static void demo2() {
	final int[] x;
	final double[] y;
	final double[] z;

	x = new int[]    {6,   7,   8,   9  };
	y = new double[] {3.0, 2.0, 1.0, 3.0};
	z = new double[] {5.0, 4.0, 4.0, 1.0};

	Swapper swapper = new Swapper() {
		public void swap(int a, int b) {
			int t1;	double t2, t3;
			t1 = x[a]; x[a] = x[b];	x[b] = t1;
			t2 = y[a]; y[a] = y[b]; y[b] = t2;
			t3 = z[a]; z[a] = z[b];	z[b] = t3;
		}
	}; 
	
	IntComparator comp = new IntComparator() {
		public int compare(int a, int b) {
			if (y[a]==y[b]) return z[a]==z[b] ? 0 : (z[a]<z[b] ? -1 : 1);
			return y[a]<y[b] ? -1 : 1;
		}
	};
	

	System.out.println("before:");
	System.out.println("X="+Arrays.toString(x));
	System.out.println("Y="+Arrays.toString(y));
	System.out.println("Z="+Arrays.toString(z));

			
	int from = 0;
	int to = x.length;
	GenericSorting.quickSort(from, to, comp, swapper);

	System.out.println("after:");
	System.out.println("X="+Arrays.toString(x));
	System.out.println("Y="+Arrays.toString(y));
	System.out.println("Z="+Arrays.toString(z));
	System.out.println("\n\n");
}
/**
 * Checks the correctness of the partition method by generating random input parameters and checking whether results are correct.
 */
public static void testRandomly(int runs) {
	org.apache.mahout.jet.random.engine.RandomEngine engine = new org.apache.mahout.jet.random.engine.MersenneTwister();
	org.apache.mahout.jet.random.Uniform gen = new org.apache.mahout.jet.random.Uniform(engine);
	
	for (int run=0; run<runs; run++) {
		int maxSize = 50;
		int maxSplittersSize = 2*maxSize;
		
		
		int size = gen.nextIntFromTo(1,maxSize);
		int from, to;
		if (size==0) { 
			from=0; to=-1;
		}
		else {
			from = gen.nextIntFromTo(0,size-1);
			to = gen.nextIntFromTo(Math.min(from,size-1),size-1);
		}

		org.apache.mahout.matrix.matrix.DoubleMatrix2D A1 = new org.apache.mahout.matrix.matrix.impl.DenseDoubleMatrix2D(size,size);
		org.apache.mahout.matrix.matrix.DoubleMatrix2D P1 = A1.viewPart(from,from,size-to,size-to);

		int intervalFrom = gen.nextIntFromTo(size/2,2*size);
		int intervalTo = gen.nextIntFromTo(intervalFrom,2*size);

		for (int i=0; i<size; i++) {
			for (int j=0; j<size; j++) {
				A1.set(i,j,gen.nextIntFromTo(intervalFrom,intervalTo));
			}
		}

		org.apache.mahout.matrix.matrix.DoubleMatrix2D A2 = A1.copy();
		org.apache.mahout.matrix.matrix.DoubleMatrix2D P2 = A2.viewPart(from,from,size-to,size-to);

		int c = 0;
		org.apache.mahout.matrix.matrix.DoubleMatrix2D S1 = org.apache.mahout.matrix.matrix.doublealgo.Sorting.quickSort.sort(P1,c);
		org.apache.mahout.matrix.matrix.DoubleMatrix2D S2 = org.apache.mahout.matrix.matrix.doublealgo.Sorting.mergeSort.sort(P2,c);

		if (!(S1.viewColumn(c).equals(S2.viewColumn(c)))) throw new InternalError();
	}

	System.out.println("All tests passed. No bug detected.");
}
}
