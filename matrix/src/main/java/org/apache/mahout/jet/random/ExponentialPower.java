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
 * Exponential Power distribution.
 * <p>
 * Valid parameter ranges: <tt>tau &gt;= 1</tt>.
 * <p>
 * Instance methods operate on a user supplied uniform random number generator; they are unsynchronized.
 * <dt>
 * Static methods operate on a default uniform random number generator; they are synchronized.
 * <p>
 * <b>Implementation:</b>
 * <dt>Method: Non-universal rejection method for logconcave densities.
 * <dt>This is a port of <tt>epd.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND / WIN-RAND</A> library.
 * C-RAND's implementation, in turn, is based upon
 * <p>
 * L. Devroye (1986): Non-Uniform Random Variate Generation , Springer Verlag, New York.
 * <p>
 *
 */
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class ExponentialPower extends AbstractContinousDistribution { 
  protected double tau;

  // cached vars for method nextDouble(tau) (for performance only)
  private double s,sm1,tau_set = -1.0;

   // The uniform random number generated shared by all <b>static</b> methods.
  protected static ExponentialPower shared = new ExponentialPower(1.0,makeDefaultGenerator());
/**
 * Constructs an Exponential Power distribution.
 * Example: tau=1.0.
 * @throws IllegalArgumentException if <tt>tau &lt; 1.0</tt>.
 */
public ExponentialPower(double tau, RandomEngine randomGenerator) {
  setRandomGenerator(randomGenerator);
  setState(tau);
}
/**
 * Returns a random number from the distribution.
 */
public double nextDouble() {
  return nextDouble(this.tau);
}
/**
 * Returns a random number from the distribution; bypasses the internal state.
 * @throws IllegalArgumentException if <tt>tau &lt; 1.0</tt>.
 */
public double nextDouble(double tau) {
  double u,u1,v,x,y;

  if (tau != tau_set) { // SET-UP 
    s = 1.0/tau;
    sm1 = 1.0 - s;

    tau_set = tau;
  }

  // GENERATOR 
  do {
    u = randomGenerator.raw();                             // U(0/1)      
    u = (2.0*u) - 1.0;                                     // U(-1.0/1.0) 
    u1 = Math.abs(u);                                      // u1=|u|     
    v = randomGenerator.raw();                             // U(0/1) 

    if (u1 <= sm1) { // Uniform hat-function for x <= (1-1/tau)   
      x = u1;
    }
    else { // Exponential hat-function for x > (1-1/tau) 
      y = tau*(1.0 - u1);                                // U(0/1) 
      x = sm1 - s*Math.log(y);
      v = v*y;
    }
  }

  // Acceptance/Rejection
  while (Math.log(v) > -Math.exp(Math.log(x)*tau));

  // Random sign 
  if (u < 0.0) 
    return x;
  else
    return -x;
}
/**
 * Sets the distribution parameter.
 * @throws IllegalArgumentException if <tt>tau &lt; 1.0</tt>.
 */
public void setState(double tau) {
  if (tau<1.0) throw new IllegalArgumentException();
  this.tau = tau;
}
/**
 * Returns a random number from the distribution.
 * @throws IllegalArgumentException if <tt>tau &lt; 1.0</tt>.
 */
public static double staticNextDouble(double tau) {
  synchronized (shared) {
    return shared.nextDouble(tau);
  }
}
/**
 * Returns a String representation of the receiver.
 */
public String toString() {
  return this.getClass().getName()+"("+tau+")";
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
