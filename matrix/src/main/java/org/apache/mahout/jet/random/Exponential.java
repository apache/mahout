/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.jet.random;

import org.apache.mahout.jet.random.engine.RandomEngine;
/**
 * Exponential Distribution (aka Negative Exponential Distribution); See the <A HREF="http://www.cern.ch/RD11/rkb/AN16pp/node78.html#SECTION000780000000000000000"> math definition</A>
 * <A HREF="http://www.statsoft.com/textbook/glose.html#Exponential Distribution"> animated definition</A>.
 * <p>
 * <tt>p(x) = lambda*exp(-x*lambda)</tt> for <tt>x &gt;= 0</tt>, <tt>lambda &gt; 0</tt>.
 * <p>
 * Instance methods operate on a user supplied uniform random number generator; they are unsynchronized.
 * <dt>
 * Static methods operate on a default uniform random number generator; they are synchronized.
 * <p>
 *
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class Exponential extends AbstractContinousDistribution { 
	protected double lambda;

 	// The uniform random number generated shared by all <b>static</b> methods.
	protected static Exponential shared = new Exponential(1.0,makeDefaultGenerator());
/**
 * Constructs a Negative Exponential distribution.
 */
public Exponential(double lambda, RandomEngine randomGenerator) {
	setRandomGenerator(randomGenerator);
	setState(lambda);
}
/**
 * Returns the cumulative distribution function.
 */
public double cdf(double x) {
	if (x <= 0.0) return 0.0;
	return 1.0 - Math.exp(-x * lambda);
}
/**
 * Returns a random number from the distribution.
 */
public double nextDouble() {
	return nextDouble(lambda);
}
/**
 * Returns a random number from the distribution; bypasses the internal state.
 */
public double nextDouble(double lambda) {
	return - Math.log(randomGenerator.raw()) / lambda;
}
/**
 * Returns the probability distribution function.
 */
public double pdf(double x) {
	if (x < 0.0) return 0.0;
	return lambda*Math.exp(-x*lambda);
}
/**
 * Sets the mean.
 */
public void setState(double lambda) {
	this.lambda = lambda;
}
/**
 * Returns a random number from the distribution with the given lambda.
 */
public static double staticNextDouble(double lambda) {
	synchronized (shared) {
		return shared.nextDouble(lambda);
	}
}
/**
 * Returns a String representation of the receiver.
 */
public String toString() {
	return this.getClass().getName()+"("+lambda+")";
}
/**
 * Sets the uniform random number generated shared by all <b>static</b> methods.
 * @param randomGenerator the new uniform random number generator to be shared.
 */
private static void xstaticSetRandomGenerator(RandomEngine randomGenerator) {
	synchronized (shared) {
		shared.setRandomGenerator(randomGenerator);
	}
}
}
