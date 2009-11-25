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
 * Hyperbolic distribution. 
 * <p>
 * Valid parameter ranges: <tt>alpha &gt; 0</tt> and <tt>beta &gt; 0</tt>.            
 * <p>
 * Instance methods operate on a user supplied uniform random number generator; they are unsynchronized.
 * <dt>
 * Static methods operate on a default uniform random number generator; they are synchronized.
 * <p>
 * <b>Implementation:</b>
 * <dt>Method: Non-Universal Rejection.
 * High performance implementation.
 * <dt>This is a port of <tt>hyplc.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND / WIN-RAND</A> library.
 * C-RAND's implementation, in turn, is based upon
 * <p>
 * L. Devroye (1986): Non-Uniform Random Variate Generation, Springer Verlag, New York.
 *
 */
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class Hyperbolic extends AbstractContinousDistribution {
  protected double alpha;
  protected double beta;

  // cached values shared for generateHyperbolic(...)
  protected double a_setup =  0.0, b_setup = -1.0;
  protected double x, u, v, e;
  protected double hr, hl, s, pm, pr, samb, pmr, mpa_1, mmb_1; 


  // The uniform random number generated shared by all <b>static</b> methods.
  protected static Hyperbolic shared = new Hyperbolic(10.0,10.0,makeDefaultGenerator());
/**
 * Constructs a Beta distribution.
 */
public Hyperbolic(double alpha, double beta, RandomEngine randomGenerator) {
  setRandomGenerator(randomGenerator);
  setState(alpha,beta);
}
/**
 * Returns a random number from the distribution.
 */
public double nextDouble() {
  return nextDouble(alpha, beta);
}
/**
 * Returns a hyperbolic distributed random number; bypasses the internal state.
 */
public double nextDouble(double alpha, double beta) {
/******************************************************************
 *                                                                *
 *        Hyperbolic Distribution - Non-Universal Rejection       *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION   : - hyplc.c samples a random number from the        *
 *                hyperbolic distribution with shape parameter a  *
 *                and b valid for a>0 and |b|<a using the         *
 *                non-universal rejection method for log-concave  *
 *                densities.                                      *
 * REFERENCE :  - L. Devroye (1986): Non-Uniform Random Variate   *
 *                Generation, Springer Verlag, New York.          *
 * SUBPROGRAM : - drand(seed) ... (0,1)-Uniform generator with    *
 *                unsigned long integer *seed.                    *
 *                                                                *
 ******************************************************************/
  double a = alpha;
  double b = beta;

  if ((a_setup != a) || (b_setup != b)) { // SET-UP 
    double mpa, mmb, mode;
    double amb;
    double a_, b_,  a_1, b_1, pl;
    double help_1, help_2;
    amb = a*a - b*b;                                        // a^2 - b^2 
    samb = Math.sqrt(amb);                                  // -log(f(mode)) 
    mode = b/samb;                                          // mode 
    help_1 = a*Math.sqrt(2.0*samb + 1.0);
    help_2 = b*(samb + 1.0);
    mpa = (help_2 + help_1)/amb;   // fr^-1(exp(-sqrt(a^2 - b^2) - 1.0)) 
    mmb = (help_2 - help_1)/amb;   // fl^-1(exp(-sqrt(a^2 - b^2) - 1.0))
    a_ =   mpa - mode;
    b_ =  -mmb + mode;
    hr = -1.0/(-a*mpa/Math.sqrt(1.0 + mpa*mpa) + b);
    hl =  1.0/(-a*mmb/Math.sqrt(1.0 + mmb*mmb) + b);
    a_1 = a_ - hr;
    b_1 = b_ - hl;
    mmb_1 = mode - b_1;                                     // lower border
    mpa_1 = mode + a_1;                                     // upper border 

    s = (a_ + b_);
    pm = (a_1 + b_1)/s;
    pr = hr/s;
    pmr = pm + pr;

    a_setup = a;
    b_setup = b;
  }

  // GENERATOR 
  for(;;)  {
    u = randomGenerator.raw();
    v = randomGenerator.raw();
    if (u <= pm) { // Rejection with a uniform majorizing function
             // over the body of the distribution 
      x = mmb_1 + u*s;
      if (Math.log(v)  <= (-a*Math.sqrt(1.0 + x*x) + b*x + samb)) break;
    }
    else {
      if (u <= pmr) {  // Rejection with an exponential envelope on the
               // right side of the mode 
        e = -Math.log((u-pm)/pr);
        x = mpa_1 + hr*e;
        if ((Math.log(v) - e) <= (-a*Math.sqrt(1.0 + x*x) + b*x + samb)) break;
      }
      else {           // Rejection with an exponential envelope on the
               // left side of the mode 
        e = Math.log((u-pmr)/(1.0 - pmr));
        x = mmb_1 + hl*e;
        if ((Math.log(v) + e) <= (-a*Math.sqrt(1.0 + x*x) + b*x + samb)) break;
      }
    }
  }

  return(x);
}
/**
 * Sets the parameters.
 */
public void setState(double alpha, double beta) {
  this.alpha = alpha;
  this.beta = beta;
}
/**
 * Returns a random number from the distribution.
 */
public static double staticNextDouble(double alpha, double beta) {
  synchronized (shared) {
    return shared.nextDouble(alpha,beta);
  }
}
/**
 * Returns a String representation of the receiver.
 */
public String toString() {
  return this.getClass().getName()+"("+alpha+","+beta+")";
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
