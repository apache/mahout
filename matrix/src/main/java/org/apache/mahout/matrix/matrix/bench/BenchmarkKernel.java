/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.colt.matrix.bench;

/**
 * Not yet documented.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 10-Nov-99
 */
class BenchmarkKernel {
/**
 * Benchmark constructor comment.
 */
protected BenchmarkKernel() {}
/**
 * Executes procedure repeatadly until more than minSeconds have elapsed.
 */
public static float run(double minSeconds, TimerProcedure procedure) {
	long iter=0;
	long minMillis = (long) (minSeconds * 1000);
	long begin=System.currentTimeMillis();
	long limit = begin + minMillis;
	while (System.currentTimeMillis() < limit) {
		procedure.init();
		procedure.apply(null);
		iter++;
	}
	long end = System.currentTimeMillis();
	if (minSeconds/iter < 0.1) { 
		// unreliable timing due to very fast iteration;
		// reading, starting and stopping timer distorts measurement
		// do it again with minimal timer overhead
		//System.out.println("iter="+iter+", minSeconds/iter="+minSeconds/iter);
		begin=System.currentTimeMillis();
		for (long i=iter; --i >= 0; ) {
			procedure.init();
			procedure.apply(null);
		}
		end = System.currentTimeMillis();
	}

	long begin2 = System.currentTimeMillis();
	int dummy=1; // prevent compiler from optimizing away the loop
	for (long i=iter; --i >= 0; ) {
		dummy *= i;
		procedure.init();
	}
	long end2 = System.currentTimeMillis();
	long elapsed = (end-begin) - (end2-begin2);
	//if (dummy != 0) throw new RuntimeException("dummy != 0");
	
	return (float) elapsed/1000.0f / iter;
}
/**
 * Returns a String with the system's properties (vendor, version, operating system, etc.)
 */
public static String systemInfo() {
	String[] properties = {
		"java.vm.vendor",
		"java.vm.version",
		"java.vm.name",
		"os.name",
		"os.version",
		"os.arch",
		"java.version",
		"java.vendor",
		"java.vendor.url"
		/*
		"java.vm.specification.version",
		"java.vm.specification.vendor",
		"java.vm.specification.name",
		"java.specification.version",
		"java.specification.vendor",
		"java.specification.name"
		*/
	};

	// build string matrix
	org.apache.mahout.colt.matrix.ObjectMatrix2D matrix = new org.apache.mahout.colt.matrix.impl.DenseObjectMatrix2D(properties.length,2);
	matrix.viewColumn(0).assign(properties);

	// retrieve property values
	for (int i=0; i<properties.length; i++) {
		String value = System.getProperty(properties[i]);
		if (value==null) value = "?"; // prop not available
		matrix.set(i,1,value);
	}

	// format matrix
	org.apache.mahout.colt.matrix.objectalgo.Formatter formatter = new org.apache.mahout.colt.matrix.objectalgo.Formatter();
	formatter.setPrintShape(false);
	return formatter.toString(matrix);
}
}
