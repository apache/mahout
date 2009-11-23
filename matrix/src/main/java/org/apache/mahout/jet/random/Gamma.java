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
import org.apache.mahout.jet.stat.Probability;
/**
 * Gamma distribution; <A HREF="http://wwwinfo.cern.ch/asdoc/shortwrupsdir/g106/top.html"> math definition</A>,
 * <A HREF="http://www.cern.ch/RD11/rkb/AN16pp/node96.html#SECTION000960000000000000000"> definition of gamma function</A>
 * and <A HREF="http://www.statsoft.com/textbook/glosf.html#Gamma Distribution"> animated definition</A>. 
 * <p>
 * <tt>p(x) = k * x^(alpha-1) * e^(-x/beta)</tt> with <tt>k = 1/(g(alpha) * b^a))</tt> and <tt>g(a)</tt> being the gamma function.
 * <p>
 * Valid parameter ranges: <tt>alpha &gt; 0</tt>.
 * <p>
 * Note: For a Gamma distribution to have the mean <tt>mean</tt> and variance <tt>variance</tt>, set the parameters as follows:
 * <pre>
 * alpha = mean*mean / variance; lambda = 1 / (variance / mean); 
 * </pre>
 * <p>
 * Instance methods operate on a user supplied uniform random number generator; they are unsynchronized.
 * <dt>
 * Static methods operate on a default uniform random number generator; they are synchronized.
 * <p>
 * <b>Implementation:</b> 
 * <dt>
 * Method: Acceptance Rejection combined with Acceptance Complement.
 * <dt>
 * High performance implementation. This is a port of <A HREF="http://wwwinfo.cern.ch/asd/lhc++/clhep/manual/RefGuide/Random/RandGamma.html">RandGamma</A> used in <A HREF="http://wwwinfo.cern.ch/asd/lhc++/clhep">CLHEP 1.4.0</A> (C++).
 * CLHEP's implementation, in turn, is based on <tt>gds.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND / WIN-RAND</A> library.
 * C-RAND's implementation, in turn, is based upon
 * <p>
 * J.H. Ahrens, U. Dieter (1974): Computer methods for sampling from gamma, beta, Poisson and binomial distributions, 
 * Computing 12, 223-246.
 * <p>
 * and
 * <p>
 * J.H. Ahrens, U. Dieter (1982): Generating gamma variates by a modified rejection technique,
 * Communications of the ACM 25, 47-54.
 *
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class Gamma extends AbstractContinousDistribution { 
	protected double alpha;
	protected double lambda;

 	// The uniform random number generated shared by all <b>static</b> methods.
	protected static Gamma shared = new Gamma(1.0,1.0,makeDefaultGenerator());
/**
 * Constructs a Gamma distribution.
 * Example: alpha=1.0, lambda=1.0.
 * @throws IllegalArgumentException if <tt>alpha &lt;= 0.0 || lambda &lt;= 0.0</tt>.
 */
public Gamma(double alpha, double lambda, RandomEngine randomGenerator) {
	setRandomGenerator(randomGenerator);
	setState(alpha,lambda);
}
/**
 * Returns the cumulative distribution function.
 */
public double cdf(double x) {
	return Probability.gamma(alpha,lambda,x);
}
/**
 * Returns a random number from the distribution.
 */
public double nextDouble() {
	return nextDouble(alpha, lambda);
}
/**
 * Returns a random number from the distribution; bypasses the internal state.
 */
public double nextDouble(double alpha, double lambda) {
/******************************************************************
 *                                                                *
 *    Gamma Distribution - Acceptance Rejection combined with     *
 *                         Acceptance Complement                  *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION:    - gds samples a random number from the standard   *
 *                gamma distribution with parameter  a > 0.       *
 *                Acceptance Rejection  gs  for  a < 1 ,          *
 *                Acceptance Complement gd  for  a >= 1 .         *
 * REFERENCES:  - J.H. Ahrens, U. Dieter (1974): Computer methods *
 *                for sampling from gamma, beta, Poisson and      *
 *                binomial distributions, Computing 12, 223-246.  *
 *              - J.H. Ahrens, U. Dieter (1982): Generating gamma *
 *                variates by a modified rejection technique,     *
 *                Communications of the ACM 25, 47-54.            *
 * SUBPROGRAMS: - drand(seed) ... (0,1)-Uniform generator with    *
 *                unsigned long integer *seed                     *
 *              - NORMAL(seed) ... Normal generator N(0,1).       *
 *                                                                *
 ******************************************************************/
 	double a = alpha;
	double aa = -1.0, aaa = -1.0, 
		b=0.0, c=0.0, d=0.0, e, r, s=0.0, si=0.0, ss=0.0, q0=0.0,
		q1 = 0.0416666664, q2 =  0.0208333723, q3 = 0.0079849875,
		q4 = 0.0015746717, q5 = -0.0003349403, q6 = 0.0003340332,
		q7 = 0.0006053049, q8 = -0.0004701849, q9 = 0.0001710320,
		a1 = 0.333333333,  a2 = -0.249999949,  a3 = 0.199999867,
		a4 =-0.166677482,  a5 =  0.142873973,  a6 =-0.124385581,
		a7 = 0.110368310,  a8 = -0.112750886,  a9 = 0.104089866,
		e1 = 1.000000000,  e2 =  0.499999994,  e3 = 0.166666848,
		e4 = 0.041664508,  e5 =  0.008345522,  e6 = 0.001353826,
		e7 = 0.000247453;

	double gds,p,q,t,sign_u,u,v,w,x;
	double v1,v2,v12;

	// Check for invalid input values

	if (a <= 0.0) throw new IllegalArgumentException(); 
	if (lambda <= 0.0) new IllegalArgumentException(); 

	if (a < 1.0) { // CASE A: Acceptance rejection algorithm gs
		b = 1.0 + 0.36788794412 * a;              // Step 1
		for(;;) {
			p = b * randomGenerator.raw();
			if (p <= 1.0) {                       // Step 2. Case gds <= 1
				gds = Math.exp(Math.log(p) / a);
				if (Math.log(randomGenerator.raw()) <= -gds) return(gds/lambda);
			}
			else {                                // Step 3. Case gds > 1
				gds = - Math.log ((b - p) / a);
				if (Math.log(randomGenerator.raw()) <= ((a - 1.0) * Math.log(gds))) return(gds/lambda);
			}
		}
	}

	else {        // CASE B: Acceptance complement algorithm gd (gaussian distribution, box muller transformation)
		if (a != aa) {                        // Step 1. Preparations
			aa = a;
			ss = a - 0.5;
			s = Math.sqrt(ss);
			d = 5.656854249 - 12.0 * s;
		}
												  // Step 2. Normal deviate
		do {
			v1 = 2.0 * randomGenerator.raw() - 1.0;
			v2 = 2.0 * randomGenerator.raw() - 1.0;
			v12 = v1*v1 + v2*v2;
		} while ( v12 > 1.0 );
		t = v1*Math.sqrt(-2.0*Math.log(v12)/v12);
		x = s + 0.5 * t;
		gds = x * x;
		if (t >= 0.0) return(gds/lambda);         // Immediate acceptance

		u = randomGenerator.raw();                // Step 3. Uniform random number
		if (d * u <= t * t * t) return(gds/lambda); // Squeeze acceptance

		if (a != aaa) {                           // Step 4. Set-up for hat case
			aaa = a;
			r = 1.0 / a;
			q0 = ((((((((q9 * r + q8) * r + q7) * r + q6) * r + q5) * r + q4) *
					  r + q3) * r + q2) * r + q1) * r;
			if (a > 3.686) {
				if (a > 13.022) {
					b = 1.77;
					si = 0.75;
					c = 0.1515 / s;
				}
			    else {
					b = 1.654 + 0.0076 * ss;
					si = 1.68 / s + 0.275;
					c = 0.062 / s + 0.024;
				}
			}
			else {
				b = 0.463 + s - 0.178 * ss;
				si = 1.235;
				c = 0.195 / s - 0.079 + 0.016 * s;
			}
		}
		if (x > 0.0) {                        // Step 5. Calculation of q
			v = t / (s + s);                  // Step 6.
			if (Math.abs(v) > 0.25) {
				q = q0 - s * t + 0.25 * t * t + (ss + ss) * Math.log(1.0 + v);
			}
			else {
				q = q0 + 0.5 * t * t * ((((((((a9 * v + a8) * v + a7) * v + a6) *
				    v + a5) * v + a4) * v + a3) * v + a2) * v + a1) * v;
			}								  // Step 7. Quotient acceptance
			if (Math.log(1.0 - u) <= q) return(gds/lambda);
		}

		for(;;) {              			      // Step 8. Double exponential deviate t
			do {
				e = -Math.log(randomGenerator.raw());
				u = randomGenerator.raw();
				u = u + u - 1.0;
				sign_u = (u > 0)? 1.0 : -1.0;
				t = b + (e * si) * sign_u;
			} while (t <= -0.71874483771719); // Step 9. Rejection of t
			v = t / (s + s);                  // Step 10. New q(t)
			if (Math.abs(v) > 0.25) {
				q = q0 - s * t + 0.25 * t * t + (ss + ss) * Math.log(1.0 + v);
			}
			else {
				q = q0 + 0.5 * t * t * ((((((((a9 * v + a8) * v + a7) * v + a6) *
				    v + a5) * v + a4) * v + a3) * v + a2) * v + a1) * v;
			}
			if (q <= 0.0) continue;           // Step 11.
			if (q > 0.5) {
				w = Math.exp(q) - 1.0;
			}
			else {
				w = ((((((e7 * q + e6) * q + e5) * q + e4) * q + e3) * q + e2) *
					     q + e1) * q;
			}                    			  // Step 12. Hat acceptance
			if ( c * u * sign_u <= w * Math.exp(e - 0.5 * t * t)) {
				x = s + 0.5 * t;
				return(x*x/lambda);
			}
		}
	}
}
/**
 * Returns the probability distribution function.
 */
public double pdf(double x) {
	if (x < 0) throw new IllegalArgumentException();
	if (x == 0) {
		if (alpha == 1.0) return 1.0/lambda;
		else return 0.0;
	}
	if (alpha == 1.0) return Math.exp(-x/lambda)/lambda;

	return Math.exp((alpha-1.0) * Math.log(x/lambda) - x/lambda - Fun.logGamma(alpha)) / lambda;
}
/**
 * Sets the mean and variance.
 * @throws IllegalArgumentException if <tt>alpha &lt;= 0.0 || lambda &lt;= 0.0</tt>.
 */
public void setState(double alpha, double lambda) {
	if (alpha <= 0.0) throw new IllegalArgumentException();
	if (lambda <= 0.0) throw new IllegalArgumentException();
	this.alpha = alpha;
	this.lambda = lambda;
}
/**
 * Returns a random number from the distribution.
 * @throws IllegalArgumentException if <tt>alpha &lt;= 0.0 || lambda &lt;= 0.0</tt>.
 */
public static double staticNextDouble(double alpha, double lambda) {
	synchronized (shared) {
		return shared.nextDouble(alpha,lambda);
	}
}
/**
 * Returns a String representation of the receiver.
 */
public String toString() {
	return this.getClass().getName()+"("+alpha+","+lambda+")";
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
