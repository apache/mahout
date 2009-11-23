/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.jet.math;

import org.apache.mahout.colt.function.IntFunction;
import org.apache.mahout.colt.function.IntIntFunction;
import org.apache.mahout.colt.function.IntIntProcedure;
import org.apache.mahout.colt.function.IntProcedure;
/** 
Integer Function objects to be passed to generic methods.
Same as {@link Functions} except operating on integers.
<p>
For aliasing see {@link #intFunctions}.

@author wolfgang.hoschek@cern.ch
@version 1.0, 09/24/99
*/
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class IntFunctions extends Object {
	/**
	Little trick to allow for "aliasing", that is, renaming this class.
	Writing code like
	<p>
	<tt>IntFunctions.chain(IntFunctions.plus,IntFunctions.mult(3),IntFunctions.chain(IntFunctions.square,IntFunctions.div(2)));</tt>
	<p>
	is a bit awkward, to say the least.
	Using the aliasing you can instead write
	<p>
	<tt>IntFunctions F = IntFunctions.intFunctions; <br>
	F.chain(F.plus,F.mult(3),F.chain(F.square,F.div(2)));</tt>
	<p>
	*/
	public static final IntFunctions intFunctions = new IntFunctions();
	
	/*****************************
	 * <H3>Unary functions</H3>
	 *****************************/
	/**
	 * Function that returns <tt>Math.abs(a) == (a < 0) ? -a : a</tt>.
	 */
	public static final IntFunction abs = new IntFunction() {
		public final int apply(int a) { return (a < 0) ? -a : a; }
	};		

	/**
	 * Function that returns <tt>a--</tt>.
	 */
	public static final IntFunction dec = new IntFunction() {
		public final int apply(int a) { return a--; }   
	};

	/**
	 * Function that returns <tt>(int) Arithmetic.factorial(a)</tt>.
	 */
	public static final IntFunction factorial = new IntFunction() {
		public final int apply(int a) { return (int) Arithmetic.factorial(a); }
	};
		
	/**
	 * Function that returns its argument.
	 */
	public static final IntFunction identity = new IntFunction() {
		public final int apply(int a) { return a; }   
	};

	/**
	 * Function that returns <tt>a++</tt>.
	 */
	public static final IntFunction inc = new IntFunction() {
		public final int apply(int a) { return a++; }   
	};

	/**
	 * Function that returns <tt>-a</tt>.
	 */
	public static final IntFunction neg = new IntFunction() {
		public final int apply(int a) { return -a; }
	};
		
	/**
	 * Function that returns <tt>~a</tt>.
	 */
	public static final IntFunction not = new IntFunction() {
		public final int apply(int a) { return ~a; }
	};
		
	/**
	 * Function that returns <tt>a < 0 ? -1 : a > 0 ? 1 : 0</tt>.
	 */
	public static final IntFunction sign = new IntFunction() {
		public final int apply(int a) { return a < 0 ? -1 : a > 0 ? 1 : 0; }
	};
		
	/**
	 * Function that returns <tt>a * a</tt>.
	 */
	public static final IntFunction square = new IntFunction() {
		public final int apply(int a) { return a * a; }
	};
		





	/*****************************
	 * <H3>Binary functions</H3>
	 *****************************/
		
	/**
	 * Function that returns <tt>a & b</tt>.
	 */
	public static final IntIntFunction and = new IntIntFunction() {
		public final int apply(int a, int b) { return a & b; }
	};
		
	/**
	 * Function that returns <tt>a < b ? -1 : a > b ? 1 : 0</tt>.
	 */
	public static final IntIntFunction compare = new IntIntFunction() {
		public final int apply(int a, int b) { return a < b ? -1 : a > b ? 1 : 0; }
	};
		
	/**
	 * Function that returns <tt>a / b</tt>.
	 */
	public static final IntIntFunction div = new IntIntFunction() {
		public final int apply(int a, int b) { return a / b; }
	};
		
	/**
	 * Function that returns <tt>a == b ? 1 : 0</tt>.
	 */
	public static final IntIntFunction equals = new IntIntFunction() {
		public final int apply(int a, int b) { return a == b ? 1 : 0; }
	};
		
	/**
	 * Function that returns <tt>a == b</tt>.
	 */
	public static final IntIntProcedure isEqual = new IntIntProcedure() {
		public final boolean apply(int a, int b) { return a == b; }
	};		

	/**
	 * Function that returns <tt>a < b</tt>.
	 */
	public static final IntIntProcedure isLess = new IntIntProcedure() {
		public final boolean apply(int a, int b) { return a < b; }
	};		

	/**
	 * Function that returns <tt>a > b</tt>.
	 */
	public static final IntIntProcedure isGreater = new IntIntProcedure() {
		public final boolean apply(int a, int b) { return a > b; }
	};		

	/**
	 * Function that returns <tt>Math.max(a,b)</tt>.
	 */
	public static final IntIntFunction max = new IntIntFunction() {
		public final int apply(int a, int b) { return (a >= b) ? a : b; }
	};
		
	/**
	 * Function that returns <tt>Math.min(a,b)</tt>.
	 */
	public static final IntIntFunction min = new IntIntFunction() {
		public final int apply(int a, int b) { return (a <= b) ? a : b; }
	};
		
	/**
	 * Function that returns <tt>a - b</tt>.
	 */
	public static final IntIntFunction minus = new IntIntFunction() {
		public final int apply(int a, int b) { return a - b; }
	};
		
	/**
	 * Function that returns <tt>a % b</tt>.
	 */
	public static final IntIntFunction mod = new IntIntFunction() {
		public final int apply(int a, int b) { return a % b; }
	};
		
	/**
	 * Function that returns <tt>a * b</tt>.
	 */
	public static final IntIntFunction mult = new IntIntFunction() {
		public final int apply(int a, int b) { return a * b; }
	};
		
	/**
	 * Function that returns <tt>a | b</tt>.
	 */
	public static final IntIntFunction or = new IntIntFunction() {
		public final int apply(int a, int b) { return a | b; }
	};
		
	/**
	 * Function that returns <tt>a + b</tt>.
	 */
	public static final IntIntFunction plus = new IntIntFunction() {
		public final int apply(int a, int b) { return a + b; }
	};
		
	/**
	 * Function that returns <tt>(int) Math.pow(a,b)</tt>.
	 */
	public static final IntIntFunction pow = new IntIntFunction() {
		public final int apply(int a, int b) { return (int) Math.pow(a,b); }
	};
	
	/**
	 * Function that returns <tt>a << b</tt>.
	 */
	public static final IntIntFunction shiftLeft = new IntIntFunction() {
		public final int apply(int a, int b) { return a << b; }
	};
		
	
	/**
	 * Function that returns <tt>a >> b</tt>.
	 */
	public static final IntIntFunction shiftRightSigned = new IntIntFunction() {
		public final int apply(int a, int b) { return a >> b; }
	};
		
	/**
	 * Function that returns <tt>a >>> b</tt>.
	 */
	public static final IntIntFunction shiftRightUnsigned = new IntIntFunction() {
		public final int apply(int a, int b) { return a >>> b; }
	};
		
	/**
	 * Function that returns <tt>a ^ b</tt>.
	 */
	public static final IntIntFunction xor = new IntIntFunction() {
		public final int apply(int a, int b) { return a ^ b; }
	};
		
/**
 * Makes this class non instantiable, but still let's others inherit from it.
 */
protected IntFunctions() {}
/**
 * Constructs a function that returns <tt>a & b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction and(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a & b; }
	};
}
/**
 * Constructs a function that returns <tt>(from<=a && a<=to) ? 1 : 0</tt>.
 * <tt>a</tt> is a variable, <tt>from</tt> and <tt>to</tt> are fixed.
 */
public static IntFunction between(final int from, final int to) {
	return new IntFunction() {
		public final int apply(int a) { return (from<=a && a<=to) ? 1 : 0; }
	};
}
/**
 * Constructs a unary function from a binary function with the first operand (argument) fixed to the given constant <tt>c</tt>.
 * The second operand is variable (free).
 * 
 * @param function a binary function taking operands in the form <tt>function.apply(c,var)</tt>.
 * @return the unary function <tt>function(c,var)</tt>.
 */
public static IntFunction bindArg1(final IntIntFunction function, final int c) {
	return new IntFunction() {
		public final int apply(int var) { return function.apply(c,var); }
	};
}
/**
 * Constructs a unary function from a binary function with the second operand (argument) fixed to the given constant <tt>c</tt>.
 * The first operand is variable (free).
 * 
 * @param function a binary function taking operands in the form <tt>function.apply(var,c)</tt>.
 * @return the unary function <tt>function(var,c)</tt>.
 */
public static IntFunction bindArg2(final IntIntFunction function, final int c) {
	return new IntFunction() {
		public final int apply(int var) { return function.apply(var,c); }
	};
}
/**
 * Constructs the function <tt>g( h(a) )</tt>.
 * 
 * @param g a unary function.
 * @param h a unary function.
 * @return the unary function <tt>g( h(a) )</tt>.
 */
public static IntFunction chain(final IntFunction g, final IntFunction h) {
	return new IntFunction() {
		public final int apply(int a) { return g.apply(h.apply(a)); }
	};
}
/**
 * Constructs the function <tt>g( h(a,b) )</tt>.
 * 
 * @param g a unary function.
 * @param h a binary function.
 * @return the unary function <tt>g( h(a,b) )</tt>.
 */
public static IntIntFunction chain(final IntFunction g, final IntIntFunction h) {
	return new IntIntFunction() {
		public final int apply(int a, int b) { return g.apply(h.apply(a,b)); }
	};
}
/**
 * Constructs the function <tt>f( g(a), h(b) )</tt>.
 * 
 * @param f a binary function.
 * @param g a unary function.
 * @param h a unary function.
 * @return the binary function <tt>f( g(a), h(b) )</tt>.
 */
public static IntIntFunction chain(final IntIntFunction f, final IntFunction g, final IntFunction h) {
	return new IntIntFunction() {
		public final int apply(int a, int b) { return f.apply(g.apply(a), h.apply(b)); }
	};
}
/**
 * Constructs a function that returns <tt>a < b ? -1 : a > b ? 1 : 0</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction compare(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a < b ? -1 : a > b ? 1 : 0; }
	};
}
/**
 * Constructs a function that returns the constant <tt>c</tt>.
 */
public static IntFunction constant(final int c) {
	return new IntFunction() {
		public final int apply(int a) { return c; }
	};
}
/**
 * Constructs a function that returns <tt>a / b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction div(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a / b; }
	};
}
/**
 * Constructs a function that returns <tt>a == b ? 1 : 0</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction equals(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a == b ? 1 : 0; }
	};
}
/**
 * Constructs a function that returns <tt>from<=a && a<=to</tt>.
 * <tt>a</tt> is a variable, <tt>from</tt> and <tt>to</tt> are fixed.
 */
public static IntProcedure isBetween(final int from, final int to) {
	return new IntProcedure() {
		public final boolean apply(int a) { return from<=a && a<=to; }
	};
}
/**
 * Constructs a function that returns <tt>a == b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntProcedure isEqual(final int b) {
	return new IntProcedure() {
		public final boolean apply(int a) { return a==b; }
	};
}
/**
 * Constructs a function that returns <tt>a > b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntProcedure isGreater(final int b) {
	return new IntProcedure() {
		public final boolean apply(int a) { return a > b; }
	};
}
/**
 * Constructs a function that returns <tt>a < b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntProcedure isLess(final int b) {
	return new IntProcedure() {
		public final boolean apply(int a) { return a < b; }
	};
}
/**
 * Constructs a function that returns <tt>Math.max(a,b)</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction max(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return (a >= b) ? a : b; }
	};
}
/**
 * Constructs a function that returns <tt>Math.min(a,b)</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction min(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return (a <= b) ? a : b; }
	};
}
/**
 * Constructs a function that returns <tt>a - b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction minus(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a - b; }
	};
}
/**
 * Constructs a function that returns <tt>a % b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction mod(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a % b; }
	};
}
/**
 * Constructs a function that returns <tt>a * b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction mult(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a * b; }
	};
}
/**
 * Constructs a function that returns <tt>a | b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction or(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a | b; }
	};
}
/**
 * Constructs a function that returns <tt>a + b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction plus(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a + b; }
	};
}
/**
 * Constructs a function that returns <tt>(int) Math.pow(a,b)</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction pow(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return (int) Math.pow(a,b); }
	};
}
/**
 * Constructs a function that returns a 32 bit uniformly distributed random number in the closed interval <tt>[Integer.MIN_VALUE,Integer.MAX_VALUE]</tt> (including <tt>Integer.MIN_VALUE</tt> and <tt>Integer.MAX_VALUE</tt>).
 * Currently the engine is {@link org.apache.mahout.jet.random.engine.MersenneTwister}
 * and is seeded with the current time.
 * <p>
 * Note that any random engine derived from {@link org.apache.mahout.jet.random.engine.RandomEngine} and any random distribution derived from {@link org.apache.mahout.jet.random.AbstractDistribution} are function objects, because they implement the proper interfaces.
 * Thus, if you are not happy with the default, just pass your favourite random generator to function evaluating methods.
 */
public static IntFunction random() {
	return new org.apache.mahout.jet.random.engine.MersenneTwister(new java.util.Date());
}
/**
 * Constructs a function that returns <tt>a << b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction shiftLeft(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a << b; }
	};
}
/**
 * Constructs a function that returns <tt>a >> b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction shiftRightSigned(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a >> b; }
	};
}
/**
 * Constructs a function that returns <tt>a >>> b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction shiftRightUnsigned(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a >>> b; }
	};
}
/**
 * Constructs a function that returns <tt>function.apply(b,a)</tt>, i.e. applies the function with the first operand as second operand and the second operand as first operand.
 * 
 * @param function a function taking operands in the form <tt>function.apply(a,b)</tt>.
 * @return the binary function <tt>function(b,a)</tt>.
 */
public static IntIntFunction swapArgs(final IntIntFunction function) {
	return new IntIntFunction() {
		public final int apply(int a, int b) { return function.apply(b,a); }
	};
}
/**
 * Constructs a function that returns <tt>a | b</tt>.
 * <tt>a</tt> is a variable, <tt>b</tt> is fixed.
 */
public static IntFunction xor(final int b) {
	return new IntFunction() {
		public final int apply(int a) { return a ^ b; }
	};
}
}
