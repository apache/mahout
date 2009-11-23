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
 * Mean-square BreitWigner distribution; See the <A HREF="http://www.cern.ch/RD11/rkb/AN16pp/node23.html#SECTION000230000000000000000"> math definition</A>.
 * <p>
 * Instance methods operate on a user supplied uniform random number generator; they are unsynchronized.
 * <dt>
 * Static methods operate on a default uniform random number generator; they are synchronized. 
 * <p>
 * <b>Implementation:</b> This is a port of <A HREF="http://wwwinfo.cern.ch/asd/lhc++/clhep/manual/RefGuide/Random/RandBreitWigner.html">RandBreitWigner</A> used in <A HREF="http://wwwinfo.cern.ch/asd/lhc++/clhep">CLHEP 1.4.0</A> (C++).
 *
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class BreitWignerMeanSquare extends BreitWigner {
	protected Uniform uniform; // helper
	
	// The uniform random number generated shared by all <b>static</b> methods.
	protected static BreitWigner shared = new BreitWignerMeanSquare(1.0,0.2,1.0,makeDefaultGenerator());
/**
 * Constructs a mean-squared BreitWigner distribution.
 * @param cut </tt>cut==Double.NEGATIVE_INFINITY</tt> indicates "don't cut".
 */
public BreitWignerMeanSquare(double mean, double gamma, double cut, RandomEngine randomGenerator) {
	super(mean,gamma,cut,randomGenerator);
	this.uniform = new Uniform(randomGenerator);
}
/**
 * Returns a deep copy of the receiver; the copy will produce identical sequences.
 * After this call has returned, the copy and the receiver have equal but separate state.
 *
 * @return a copy of the receiver.
 */
public Object clone() {
	BreitWignerMeanSquare copy = (BreitWignerMeanSquare) super.clone();
	if (this.uniform != null) copy.uniform = new Uniform(copy.randomGenerator);
	return copy;
}
/**
 * Returns a mean-squared random number from the distribution; bypasses the internal state.
 * @param cut </tt>cut==Double.NEGATIVE_INFINITY</tt> indicates "don't cut".
 */
public double nextDouble(double mean,double gamma,double cut) {
	if (gamma == 0.0) return mean;
	if (cut==Double.NEGATIVE_INFINITY) { // don't cut
		double val = Math.atan(-mean/gamma);
		double rval = this.uniform.nextDoubleFromTo(val, Math.PI/2.0);
		double displ = gamma*Math.tan(rval);
		return Math.sqrt(mean*mean + mean*displ);
	}
	else {
		double tmp = Math.max(0.0,mean-cut);
		double lower = Math.atan( (tmp*tmp-mean*mean)/(mean*gamma) );
		double upper = Math.atan( ((mean+cut)*(mean+cut)-mean*mean)/(mean*gamma) );
		double rval = this.uniform.nextDoubleFromTo(lower, upper);

		double displ = gamma*Math.tan(rval);
		return Math.sqrt(Math.max(0.0, mean*mean + mean*displ));
	}
}
/**
 * Returns a random number from the distribution.
 * @param cut </tt>cut==Double.NEGATIVE_INFINITY</tt> indicates "don't cut".
 */
public static double staticNextDouble(double mean,double gamma,double cut) {
	synchronized (shared) {
		return shared.nextDouble(mean,gamma,cut);
	}
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
