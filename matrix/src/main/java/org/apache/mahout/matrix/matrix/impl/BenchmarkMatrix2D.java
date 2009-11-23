/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.colt.matrix.impl;

import org.apache.mahout.colt.matrix.DoubleMatrix2D;
/**
Benchmarks the performance of matrices. Here are the results of some encouraging 
measurements. Note that all benchmarks only measure the time spent in accessing 
a matrix element; they exclude the loop itself. 
<p> 
<center>
  <table border cellpadding="3" cellspacing="0" align="center">
	<tr valign="middle" bgcolor="#33CC66" nowrap align="center"> 
	  <td nowrap colspan="7"> <font size="+2">Iteration Performance [million method 
		calls per second]</font><br>
		<font size="-1">Pentium Pro 200 Mhz, SunJDK 1.2.2, NT, java -classic,<br>
		60 times repeating the same iteration </font></td>
	</tr>
	<tr valign="middle" bgcolor="#33CC66" nowrap align="center"> 
	  <td nowrap> 
		<div align="left"> Element type</div>
	  </td>
	  <td nowrap colspan="6"> Matrix2D type </td>
	</tr>
	<tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> 
	  <td nowrap bgcolor="#FF9966" rowspan="2"> 
		<div align="left"> .</div>
	  </td>
	  <td bgcolor="#FF9966" colspan="2"> 
		<p><tt>DenseDoubleMatrix2D</tt><br>
		  1000 x 1000 </p>
	  </td>
	  <td bgcolor="#FF9966" colspan="2">&nbsp;</td>
	  <td bgcolor="#FF9966" colspan="2"> 
		<p><tt>SparseDoubleMatrix2D</tt><br>
		  100 x 1000,<br>
		  <font size="-1"> minLoadFactor=0.2, maxLoadFactor=0.5, initialCapacity 
		  = 0</font></p>
	  </td>
	</tr>
	<tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> 
	  <td bgcolor="#FF9966"> getQuick</td>
	  <td bgcolor="#FF9966"> setQuick</td>
	  <td bgcolor="#FF9966">&nbsp;</td>
	  <td bgcolor="#FF9966">&nbsp;</td>
	  <td bgcolor="#FF9966"> getQuick</td>
	  <td bgcolor="#FF9966">setQuick</td>
	</tr>
	<tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> 
	  <td nowrap bgcolor="#FF9966">double</td>
	  <td nowrap>5</td>
	  <td nowrap>5</td>
	  <td nowrap>&nbsp;</td>
	  <td nowrap>&nbsp;</td>
	  <td nowrap>1</td>
	  <td nowrap>0.27</td>
	</tr>
	<tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> 
	  <td nowrap bgcolor="#FF9966"> int</td>
	  <td nowrap>5 </td>
	  <td nowrap>5.5 </td>
	  <td nowrap>&nbsp;</td>
	  <td nowrap>&nbsp;</td>
	  <td nowrap>1 </td>
	  <td nowrap>0.3</td>
	</tr>
  </table>
</center>
<p align="left"> As can be seen, sparse matrices are certainly not quite as quick 
  as dense ones, but not really slow either. Considering their minimal footprint 
  they can be a real alternative. 
<p> Comparing the OO abstractions to zero-abstraction primitive Java arrays may 
  or may not be useful. Still, the table below provides some interesting information. 
  For example, access to <tt>Type_T_Matrix2D</tt> is quicker than naive usage 
  of <tt>Type_T_[]</tt>. Primitive arrays should only be considered if the optimized 
  form can be applied. Note again that all benchmarks only measure the time spent 
  in accessing a matrix element; they exclude the loop itself. 
<p> 
<center>
  <table border cellpadding="3" cellspacing="0" align="center" width="617">
	<tr valign="middle" bgcolor="#33CC66" nowrap align="center"> 
	  <td height="30" nowrap colspan="7"> <font size="+2">Iteration Performance 
		[million element accesses per second]</font><br>
		<font size="-1">Pentium Pro 200 Mhz, SunJDK 1.2.2, NT, java -classic,<br>
		200 times repeating the same iteration </font></td>
	</tr>
	<tr valign="middle" bgcolor="#33CC66" nowrap align="center"> 
	  <td width="78" height="30" nowrap> 
		<div align="left"> Element type</div>
	  </td>
	  <td height="30" nowrap colspan="6"> 
		<div align="center">Matrix2D type = Java array <tt>double[][]</tt></div>
	  </td>
	</tr>
	<tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> 
	  <td width="78" height="60" nowrap bgcolor="#FF9966" rowspan="2"> 
		<div align="left"> .</div>
	  </td>
	  <td height="132" bgcolor="#FF9966" colspan="2"> 
		<p>Unoptimized Form<br>
		  1000 x 1000<br>
		<div align="left"> <font size="-1"> 
		  <pre>
for (int row=0; row < rows; row++) { 
   for (int col=0; col < columns; ) { 
	  value = m[row][col++];
	  ...
   }
}
</pre>
		  </font> </div>
	  </td>
	  <td height="132" bgcolor="#FF9966" colspan="4"> Optimized Form<br>
		1000 x 1000 
		<div align="left"> <font size="-1"> 
		  <pre>
for (int row=0; row < rows; row++) { 
   int[] r = matrix[row]; 
   for (int col=0; col < columns; ) { 
	  value = r[col++];
	  ...
   }
}
</pre>
		  </font> </div>
	  </td>
	</tr>
	<tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> 
	  <td width="152" height="30" bgcolor="#FF9966"> getting</td>
	  <td width="144" height="30" bgcolor="#FF9966"> setting</td>
	  <td width="150" height="30" bgcolor="#FF9966"> getting</td>
	  <td width="138" height="30" bgcolor="#FF9966" colspan="3"> setting</td>
	</tr>
	<tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> 
	  <td width="78" height="30" nowrap bgcolor="#FF9966">double</td>
	  <td width="152" height="30" nowrap>1.6</td>
	  <td width="144" height="30" nowrap>1.8</td>
	  <td width="150" height="30" nowrap>18</td>
	  <td width="138" height="30" nowrap colspan="3">11</td>
	</tr>
	<tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> 
	  <td width="78" height="30" nowrap bgcolor="#FF9966"> int</td>
	  <td width="152" height="30" nowrap>1.5 </td>
	  <td width="144" height="30" nowrap>1.8</td>
	  <td width="150" height="30" nowrap>28</td>
	  <td width="138" height="30" nowrap colspan="3">26</td>
	</tr>
  </table>
</center>
<left> 

@author wolfgang.hoschek@cern.ch
@version 1.0, 09/24/99
*/
class BenchmarkMatrix2D {
/**
 * Makes this class non instantiable, but still let's others inherit from it.
 */
protected BenchmarkMatrix2D() {
	throw new RuntimeException("Non instantiable");
}
/**
 * Runs a bench on matrices holding double elements.
 */
public static void doubleBenchmark(int runs, int rows, int columns, String kind, boolean print, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
	System.out.println("benchmarking double matrix");
	// certain loops need to be constructed so that the jitter can't optimize them away and we get fantastic numbers.
	// this involves primarly read-loops

	org.apache.mahout.colt.Timer timer1 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer2 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer3 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer4 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop2 = new org.apache.mahout.colt.Timer();

	emptyLoop.start();
	int dummy = 0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy++;
			}
		}
	}
	emptyLoop.stop();
	System.out.println(dummy); // !!! so that the jitter can't optimize away the whole loop
	
	emptyLoop2.start();
	dummy = 3;
	double dummy2 = 0;
	for (int i=0; i<runs; i++) {
		for (int value = 0, column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy2 += dummy;
			}
		}
	}
	emptyLoop2.stop();
	System.out.println(dummy2); // !!! so that the jitter can't optimize away the whole loop
	

	long before = Runtime.getRuntime().freeMemory();
	long size = (((long)rows)*columns)*runs;

	DoubleMatrix2D  matrix = null;
	if (kind.equals("sparse")) matrix = new SparseDoubleMatrix2D(rows,columns,initialCapacity,minLoadFactor,maxLoadFactor);
	else if (kind.equals("dense")) matrix = new DenseDoubleMatrix2D(rows,columns);
	//else if (kind.equals("denseArray")) matrix = new DoubleArrayMatrix2D(rows,columns);
	else throw new RuntimeException("unknown kind");
	
	System.out.println("\nNow filling...");
	//if (kind.equals("sparse")) ((SparseDoubleMatrix2D)matrix).elements.hashCollisions = 0;
	for (int i=0; i<runs; i++) {
		matrix.assign(0);
		matrix.ensureCapacity(initialCapacity);
		if (kind.equals("sparse")) ((SparseDoubleMatrix2D)matrix).ensureCapacity(initialCapacity);
		timer1.start();
		int value = 0;
		for (int row=0; row < rows; row++) {
			for (int column=0; column < columns; column++) {
				matrix.setQuick(row,column,value++);
			}
		}
		timer1.stop();
	}
	timer1.display();
	timer1.minus(emptyLoop).display();
	System.out.println(size / timer1.minus(emptyLoop).seconds() +" elements / sec");
	
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	long after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after) / 1024);
	System.out.println("bytes needed per non-zero="+(before-after) / (double)matrix.cardinality());
	if (print) {
		System.out.println(matrix);
		if (kind.equals("sparse")) System.out.println("map="+((SparseDoubleMatrix2D)matrix).elements);
	}
	/*
	if (kind.equals("sparse")) {
		int hashCollisions = ((SparseDoubleMatrix2D)matrix).elements.hashCollisions;
		System.out.println("hashCollisions="+hashCollisions);
		System.out.println("--> "+ ((double)hashCollisions / (rows*columns)) +" hashCollisions/element on average.");
	}
	*/
	
	System.out.println("\nNow reading...");
	//if (kind.equals("sparse")) ((SparseDoubleMatrix2D)matrix).elements.hashCollisions = 0;
	timer2.start();
	double element=0;
	for (int i=0; i<runs; i++) {
		for (int row=0; row < rows; row++) {
			for (int column=0; column < columns; column++) {
				element += matrix.getQuick(row,column);
			}
		}		
	}
	timer2.stop().display();	
	timer2.minus(emptyLoop2).display();
	System.out.println(size / timer2.minus(emptyLoop2).seconds() +" elements / sec");
	if (print) System.out.println(matrix);
	//if (kind.equals("sparse")) System.out.println("hashCollisions="+((SparseDoubleMatrix2D)matrix).elements.hashCollisions);
	System.out.println(element); // !!! so that the jitter can't optimize away the whole loop
	
	System.out.println("\nNow reading view...");
	DoubleMatrix2D view = matrix.viewPart(0,0,rows,columns);
	timer4.start();
	element=0;
	for (int i=0; i<runs; i++) {
		for (int row=0; row < rows; row++) {
			for (int column=0; column < columns; column++) {
				element += view.getQuick(row,column);
			}
		}		
	}
	timer4.stop().display();	
	timer4.minus(emptyLoop2).display();
	System.out.println(size / timer4.minus(emptyLoop2).seconds() +" elements / sec");
	if (print) System.out.println(view);
	//if (kind.equals("sparse")) System.out.println("hashCollisions="+((SparseDoubleMatrix2D)view).elements.hashCollisions);
	System.out.println(element); // !!! so that the jitter can't optimize away the whole loop
	
	System.out.println("\nNow removing...");
	before = Runtime.getRuntime().freeMemory();
	//if (kind.equals("sparse")) ((SparseDoubleMatrix2D)matrix).elements.hashCollisions = 0;
	for (int i=0; i<runs; i++) {
		// initializing
		for (int row=0; row < rows; row++) {
			for (int column=0; column < columns; column++) {
				matrix.setQuick(row,column,1);
			}
		}
		timer3.start();
		for (int row=0; row < rows; row++) {
			for (int column=0; column < columns; column++) {
				matrix.setQuick(row,column,0);
			}
		}		
		timer3.stop();
	}
	timer3.display();
	timer3.minus(emptyLoop).display();
	System.out.println(size / timer3.minus(emptyLoop).seconds() +" elements / sec");
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after)/1024);
	System.out.println("KB free="+(after/1024));
	
	if (print) System.out.println(matrix);
	//if (kind.equals("sparse")) System.out.println("hashCollisions"+((SparseDoubleMatrix2D)matrix).elements.hashCollisions);


	System.out.println("bye bye.");
}
/**
 * Runs a bench on matrices holding double elements.
 */
public static void doubleBenchmarkMult(int runs, int rows, int columns, String kind, boolean print, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
	System.out.println("benchmarking double matrix");
	// certain loops need to be constructed so that the jitter can't optimize them away and we get fantastic numbers.
	// this involves primarly read-loops

	org.apache.mahout.colt.Timer timer1 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer2 = new org.apache.mahout.colt.Timer();

	long size = (((long)rows)*columns)*runs;

	DoubleMatrix2D  matrix = null;
	if (kind.equals("sparse")) matrix = new SparseDoubleMatrix2D(rows,columns,initialCapacity,minLoadFactor,maxLoadFactor);
	else if (kind.equals("dense")) matrix = new DenseDoubleMatrix2D(rows,columns);
	//else if (kind.equals("denseArray")) matrix = new DoubleArrayMatrix2D(rows,columns);
	else throw new RuntimeException("unknown kind");
	
	System.out.println("\nNow multiplying...");
	matrix.assign(1);
	//if (kind.equals("sparse")) ((SparseDoubleMatrix2D)matrix).elements.hashCollisions = 0;
	for (int i=0; i<runs; i++) {
		timer1.start();
		org.apache.mahout.colt.matrix.doublealgo.Transform.mult(matrix, 3);
		timer1.stop();
	}
	timer1.display();
	System.out.println(size / timer1.seconds() +" elements / sec");
	
	if (print) {
		System.out.println(matrix);
	}
	/*
	if (kind.equals("sparse")) {
		int hashCollisions = ((SparseDoubleMatrix2D)matrix).elements.hashCollisions;
		System.out.println("hashCollisions="+hashCollisions);
		System.out.println("--> "+ ((double)hashCollisions / (rows*columns)) +" hashCollisions/element on average.");
	}
	*/
	
	System.out.println("\nNow multiplying2...");
	matrix.assign(1);
	//if (kind.equals("sparse")) ((SparseDoubleMatrix2D)matrix).elements.hashCollisions = 0;
	for (int i=0; i<runs; i++) {
		timer2.start();
		org.apache.mahout.colt.matrix.doublealgo.Transform.mult(matrix,3);
		timer2.stop();
	}
	timer2.display();
	System.out.println(size / timer2.seconds() +" elements / sec");
	
	if (print) {
		System.out.println(matrix);
	}
	/*
	if (kind.equals("sparse")) {
		int hashCollisions = ((SparseDoubleMatrix2D)matrix).elements.hashCollisions;
		System.out.println("hashCollisions="+hashCollisions);
		System.out.println("--> "+ ((double)hashCollisions / (rows*columns)) +" hashCollisions/element on average.");
	}
	*/
	System.out.println("bye bye.");
}
/**
 * Runs a bench on matrices holding double elements.
 */
public static void doubleBenchmarkPrimitive(int runs, int rows, int columns, boolean print) {
	// certain loops need to be constructed so that the jitter can't optimize them away and we get fantastic numbers.
	// this involves primarly read-loops
	org.apache.mahout.colt.Timer timer1 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer2 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer3 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop2 = new org.apache.mahout.colt.Timer();

	emptyLoop.start();
	int dummy = 0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy++;
			}
		}
	}
	emptyLoop.stop();
	System.out.println(dummy); // !!! so that the jitter can't optimize away the whole loop
	
	emptyLoop2.start();
	dummy = 3;
	double dummy2 = 0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy2 += dummy;
			}
		}
	}
	emptyLoop2.stop();
	System.out.println(dummy2); // !!! so that the jitter can't optimize away the whole loop

	long before = Runtime.getRuntime().freeMemory();
	long size = (((long)rows)*columns)*runs;

	double[][] matrix = new double[rows][columns];
	
	System.out.println("\nNow filling...");
	for (int i=0; i<runs; i++) {
		timer1.start();
		int value = 0;
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				matrix[row][column] = value++;
			}
		}
		timer1.stop();
	}
	timer1.display();
	timer1.minus(emptyLoop).display();
	System.out.println(size / timer1.minus(emptyLoop).seconds() +" elements / sec");
	
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	long after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after) / 1024);
	if (print) {
		DenseDoubleMatrix2D m = new DenseDoubleMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}
	
	System.out.println("\nNow reading...");
	timer2.start();
	double element=0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				element += matrix[row][column]; 
			}
		}		
	}
	timer2.stop().display();	
	timer2.minus(emptyLoop2).display();
	System.out.println(size / timer2.minus(emptyLoop2).seconds() +" elements / sec");
	if (print) {
		DenseDoubleMatrix2D m = new DenseDoubleMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}
	System.out.println(element); // !!! so that the jitter can't optimize away the whole loop
	
	System.out.println("\nNow removing...");
	before = Runtime.getRuntime().freeMemory();
	for (int i=0; i<runs; i++) {
		/*
		// initializing
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				matrix.setQuick(row,column,1);
			}
		}
		*/
		timer3.start();
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				matrix[row][column] = 0;
			}
		}		
		timer3.stop();
	}
	timer3.display();
	timer3.minus(emptyLoop).display();
	System.out.println(size / timer3.minus(emptyLoop).seconds() +" elements / sec");
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after)/1024);
	System.out.println("KB free="+(after/1024));
	
	if (print) {
		DenseDoubleMatrix2D m = new DenseDoubleMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}

	System.out.println("bye bye.");
}
/**
 * Runs a bench on matrices holding double elements.
 */
public static void doubleBenchmarkPrimitiveOptimized(int runs, int rows, int columns, boolean print) {
	// certain loops need to be constructed so that the jitter can't optimize them away and we get fantastic numbers.
	// this involves primarly read-loops
	org.apache.mahout.colt.Timer timer1 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer2 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer3 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop2 = new org.apache.mahout.colt.Timer();

	emptyLoop.start();
	int dummy = 0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy++;
			}
		}
	}
	emptyLoop.stop();
	System.out.println(dummy); // !!! so that the jitter can't optimize away the whole loop
	
	emptyLoop2.start();
	dummy = 3;
	double dummy2 = 0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy2 += dummy;
			}
		}
	}
	emptyLoop2.stop();
	System.out.println(dummy2); // !!! so that the jitter can't optimize away the whole loop

	long before = Runtime.getRuntime().freeMemory();
	long size = (((long)rows)*columns)*runs;

	double[][] matrix = new double[rows][columns];
	
	System.out.println("\nNow filling...");
	for (int i=0; i<runs; i++) {
		timer1.start();
		int value = 0;
		for (int row=0; row < rows; row++) {
			double[] r = matrix[row];
			for (int column=0; column < columns; column++) {
				r[column] = value++;
				//matrix[row][column] = value++;
			}
		}
		timer1.stop();
	}
	timer1.display();
	timer1.minus(emptyLoop).display();
	System.out.println(size / timer1.minus(emptyLoop).seconds() +" elements / sec");
	
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	long after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after) / 1024);
	if (print) {
		DenseDoubleMatrix2D m = new DenseDoubleMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}
	
	System.out.println("\nNow reading...");
	timer2.start();
	double element=0;
	for (int i=0; i<runs; i++) {
		for (int row=0; row < rows; row++) {
			double[] r = matrix[row];
			for (int column=0; column < columns; column++) {
				element += r[column];
				//element += matrix[row][column]; 
			}
		}		
	}
	timer2.stop().display();	
	timer2.minus(emptyLoop2).display();
	System.out.println(size / timer2.minus(emptyLoop2).seconds() +" elements / sec");
	if (print) {
		DenseDoubleMatrix2D m = new DenseDoubleMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}
	System.out.println(element); // !!! so that the jitter can't optimize away the whole loop
	
	System.out.println("\nNow removing...");
	before = Runtime.getRuntime().freeMemory();
	for (int i=0; i<runs; i++) {
		timer3.start();
		for (int row=0; row < rows; row++) {
			double[] r = matrix[row];
			for (int column=0; column < columns; column++) {
				r[column] = 0;
				//matrix[row][column] = 0;
			}
		}		
		timer3.stop();
	}
	timer3.display();
	timer3.minus(emptyLoop).display();
	System.out.println(size / timer3.minus(emptyLoop).seconds() +" elements / sec");
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after)/1024);
	System.out.println("KB free="+(after/1024));
	
	if (print) {
		DenseDoubleMatrix2D m = new DenseDoubleMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}

	System.out.println("bye bye.");
}
/**
 * Runs a bench on matrices holding int elements.
 */
public static void intBenchmark(int runs, int rows, int columns, String kind, boolean print, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
	throw new InternalError();
	/*
	// certain loops need to be constructed so that the jitter can't optimize them away and we get fantastic numbers.
	// this involves primarly read-loops

	org.apache.mahout.colt.Timer timer1 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer2 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer3 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop2 = new org.apache.mahout.colt.Timer();

	emptyLoop.start();
	int dummy = 0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy++;
			}
		}
	}
	emptyLoop.stop();
	System.out.println(dummy); // !!! so that the jitter can't optimize away the whole loop
	
	emptyLoop2.start();
	dummy = 3;
	int dummy2 = 0;
	for (int i=0; i<runs; i++) {
		for (int value = 0, column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy2 += dummy;
			}
		}
	}
	emptyLoop2.stop();
	System.out.println(dummy2); // !!! so that the jitter can't optimize away the whole loop

	long before = Runtime.getRuntime().freeMemory();
	long size = (((long)rows)*columns)*runs;

	AbstractIntMatrix2D  matrix = null;
	if (kind.equals("sparse")) matrix = new SparseIntMatrix2D(rows,columns,initialCapacity,minLoadFactor,maxLoadFactor);
	else if (kind.equals("dense")) matrix = new DenseIntMatrix2D(rows,columns);
	//else if (kind.equals("denseArray")) matrix = new DoubleArrayMatrix2D(rows,columns);
	else throw new RuntimeException("unknown kind");
	
	System.out.println("\nNow filling...");
	if (kind.equals("sparse")) ((SparseIntMatrix2D)matrix).elements.hashCollisions = 0;
	for (int i=0; i<runs; i++) {
		matrix.assign(0);
		matrix.ensureCapacity(initialCapacity);
		if (kind.equals("sparse")) ((SparseIntMatrix2D)matrix).ensureCapacity(initialCapacity);
		timer1.start();
		int value = 0;
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				matrix.setQuick(row,column,value++);
			}
		}
		timer1.stop();
	}
	timer1.display();
	timer1.minus(emptyLoop).display();
	System.out.println(size / timer1.minus(emptyLoop).seconds() +" elements / sec");
	
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	long after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after) / 1024);
	System.out.println("bytes needed per non-zero="+(before-after) / (double)matrix.cardinality());
	if (print) {
		System.out.println(matrix);
		if (kind.equals("sparse")) System.out.println("map="+((SparseIntMatrix2D)matrix).elements);
	}
	if (kind.equals("sparse")) {
		int hashCollisions = ((SparseIntMatrix2D)matrix).elements.hashCollisions;
		System.out.println("hashCollisions="+hashCollisions);
		System.out.println("--> "+ ((double)hashCollisions / (rows*columns)) +" probes/element on average.");
	}
	
	System.out.println("\nNow reading...");
	if (kind.equals("sparse")) ((SparseIntMatrix2D)matrix).elements.hashCollisions = 0;
	timer2.start();
	int element=0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				element += matrix.getQuick(row,column);
			}
		}		
	}
	timer2.stop().display();	
	timer2.minus(emptyLoop2).display();
	System.out.println(size / timer2.minus(emptyLoop2).seconds() +" elements / sec");
	if (print) System.out.println(matrix);
	if (kind.equals("sparse")) System.out.println("hashCollisions="+((SparseIntMatrix2D)matrix).elements.hashCollisions);
	System.out.println(element); // !!! so that the jitter can't optimize away the whole loop
	
	System.out.println("\nNow removing...");
	before = Runtime.getRuntime().freeMemory();
	if (kind.equals("sparse")) ((SparseIntMatrix2D)matrix).elements.hashCollisions = 0;
	for (int i=0; i<runs; i++) {
		// initializing
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				matrix.setQuick(row,column,1);
			}
		}
		timer3.start();
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				matrix.setQuick(row,column,0);
			}
		}		
		timer3.stop();
	}
	timer3.display();
	timer3.minus(emptyLoop).display();
	System.out.println(size / timer3.minus(emptyLoop).seconds() +" elements / sec");
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after)/1024);
	System.out.println("KB free="+(after/1024));
	
	if (print) System.out.println(matrix);
	if (kind.equals("sparse")) System.out.println("hashCollisions="+((SparseIntMatrix2D)matrix).elements.hashCollisions);


	System.out.println("bye bye.");
	*/
}
/**
 * Runs a bench on matrices holding int elements.
 */
public static void intBenchmarkPrimitive(int runs, int rows, int columns, boolean print) {
	throw new InternalError();
	/*
	// certain loops need to be constructed so that the jitter can't optimize them away and we get fantastic numbers.
	// this involves primarly read-loops
	org.apache.mahout.colt.Timer timer1 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer2 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer3 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop2 = new org.apache.mahout.colt.Timer();

	emptyLoop.start();
	int dummy = 0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy++;
			}
		}
	}
	emptyLoop.stop();
	System.out.println(dummy); // !!! so that the jitter can't optimize away the whole loop
	
	emptyLoop2.start();
	dummy = 3;
	int dummy2 = 0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy2 += dummy;
			}
		}
	}
	emptyLoop2.stop();
	System.out.println(dummy2); // !!! so that the jitter can't optimize away the whole loop

	long before = Runtime.getRuntime().freeMemory();
	long size = (((long)rows)*columns)*runs;

	int[][] matrix = new int[rows][columns];
	
	System.out.println("\nNow filling...");
	for (int i=0; i<runs; i++) {
		timer1.start();
		int value = 0;
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				matrix[row][column] = value++;
			}
		}
		timer1.stop();
	}
	timer1.display();
	timer1.minus(emptyLoop).display();
	System.out.println(size / timer1.minus(emptyLoop).seconds() +" elements / sec");
	
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	long after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after) / 1024);
	if (print) {
		DenseIntMatrix2D m = new DenseIntMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}
	
	System.out.println("\nNow reading...");
	timer2.start();
	int element=0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				element += matrix[row][column]; 
			}
		}		
	}
	timer2.stop().display();	
	timer2.minus(emptyLoop2).display();
	System.out.println(size / timer2.minus(emptyLoop2).seconds() +" elements / sec");
	if (print) {
		DenseIntMatrix2D m = new DenseIntMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}
	System.out.println(element); // !!! so that the jitter can't optimize away the whole loop
	
	System.out.println("\nNow removing...");
	before = Runtime.getRuntime().freeMemory();
	for (int i=0; i<runs; i++) {
		timer3.start();
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				matrix[row][column] = 0;
			}
		}		
		timer3.stop();
	}
	timer3.display();
	timer3.minus(emptyLoop).display();
	System.out.println(size / timer3.minus(emptyLoop).seconds() +" elements / sec");
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after)/1024);
	System.out.println("KB free="+(after/1024));
	
	if (print) {
		DenseIntMatrix2D m = new DenseIntMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}

	System.out.println("bye bye.");
	*/
}
/**
 * Runs a bench on matrices holding int elements.
 */
public static void intBenchmarkPrimitiveOptimized(int runs, int rows, int columns, boolean print) {
	throw new InternalError();
	/*
	// certain loops need to be constructed so that the jitter can't optimize them away and we get fantastic numbers.
	// this involves primarly read-loops
	org.apache.mahout.colt.Timer timer1 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer2 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer timer3 = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop = new org.apache.mahout.colt.Timer();
	org.apache.mahout.colt.Timer emptyLoop2 = new org.apache.mahout.colt.Timer();

	emptyLoop.start();
	int dummy = 0;
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy++;
			}
		}
	}
	emptyLoop.stop();
	System.out.println(dummy); // !!! so that the jitter can't optimize away the whole loop
	
	int[][] matrix = new int[rows][columns];
	emptyLoop2.start();
	dummy = 3;
	int dummy2 = 7;
	System.out.println(dummy2); // !!! so that the jitter can't optimize away the whole loop
	for (int i=0; i<runs; i++) {
		for (int column=0; column < columns; column++) {
			for (int row=0; row < rows; row++) {
				dummy2 += dummy; //matrix[row][column];
			}
		}
	}
	emptyLoop2.stop();
	System.out.println(dummy2); // !!! so that the jitter can't optimize away the whole loop

	long before = Runtime.getRuntime().freeMemory();
	long size = (((long)rows)*columns)*runs;

	
	System.out.println("\nNow filling...");
	for (int i=0; i<runs; i++) {
		timer1.start();
		int value = 0;
		for (int row=0; row < rows; row++) {
			int[] r = matrix[row];
			for (int column=0; column < columns; column++) {
				r[column] = value++;
				//matrix[row][column] = value++;
			}
		}
		timer1.stop();
	}
	timer1.display();
	timer1.minus(emptyLoop).display();
	System.out.println(size / timer1.minus(emptyLoop).seconds() +" elements / sec");
	
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	long after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after) / 1024);
	if (print) {
		DenseIntMatrix2D m = new DenseIntMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}
	
	System.out.println("\nNow reading...");
	timer2.start();
	int element=0;
	for (int i=0; i<runs; i++) {
		for (int row=0; row < rows; row++) {
			int[] r = matrix[row];
			for (int column=0; column < columns; column++) {
				element += r[column];
				//element += matrix[row][column]; 
			}
		}		
	}
	timer2.stop().display();	
	timer2.minus(emptyLoop2).display();
	System.out.println(size / timer2.minus(emptyLoop2).seconds() +" elements / sec");
	if (print) {
		DenseIntMatrix2D m = new DenseIntMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}
	System.out.println(element); // !!! so that the jitter can't optimize away the whole loop
	
	System.out.println("\nNow removing...");
	before = Runtime.getRuntime().freeMemory();
	for (int i=0; i<runs; i++) {
		timer3.start();
		for (int row=0; row < rows; row++) {
			int[] r = matrix[row];
			for (int column=0; column < columns; column++) {
				r[column] = 0;
				//matrix[row][column] = 0;
			}
		}		
		timer3.stop();
	}
	timer3.display();
	timer3.minus(emptyLoop).display();
	System.out.println(size / timer3.minus(emptyLoop).seconds() +" elements / sec");
	Runtime.getRuntime().gc(); // invite gc
	try { Thread.currentThread().sleep(1000); } catch (InterruptedException exc) {};
	after = Runtime.getRuntime().freeMemory();
	System.out.println("KB needed="+(before-after)/1024);
	System.out.println("KB free="+(after/1024));
	
	if (print) {
		DenseIntMatrix2D m = new DenseIntMatrix2D(rows,columns);
		m.assign(matrix);
		System.out.println(m);
	}

	System.out.println("bye bye.");
	*/
}
/**
 * Benchmarks various methods of this class.
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
	String type = args[8];
	String command = args[9];
	
	if (type.equals("int")) {
		if (kind.equals("primitive")) intBenchmarkPrimitive(runs,rows,columns,print);
		else if (kind.equals("primitiveOpt")) intBenchmarkPrimitiveOptimized(runs,rows,columns,print);
		else intBenchmark(runs,rows,columns,kind,print,initialCapacity,minLoadFactor,maxLoadFactor);
	}
	else if (type.equals("double")) {
		if (kind.equals("primitive")) doubleBenchmarkPrimitive(runs,rows,columns,print);
		else if (kind.equals("primitiveOpt")) doubleBenchmarkPrimitiveOptimized(runs,rows,columns,print);
		else if (command.equals("mult")) doubleBenchmarkMult(runs,rows,columns,kind,print,initialCapacity,minLoadFactor,maxLoadFactor);
		else doubleBenchmark(runs,rows,columns,kind,print,initialCapacity,minLoadFactor,maxLoadFactor);
	}
	 
}
}
