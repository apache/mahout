/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.jet.random.engine.RandomEngine;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public abstract class AbstractDistribution extends PersistentObject
    implements org.apache.mahout.math.function.DoubleFunction, org.apache.mahout.math.function.IntFunction {

  protected RandomEngine randomGenerator;

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected AbstractDistribution() {
  }

  /**
   * Equivalent to <tt>nextDouble()</tt>. This has the effect that distributions can now be used as function objects,
   * returning a random number upon function evaluation.
   */
  @Override
  public double apply(double dummy) {
    return nextDouble();
  }

  /**
   * Equivalent to <tt>nextInt()</tt>. This has the effect that distributions can now be used as function objects,
   * returning a random number upon function evaluation.
   */
  @Override
  public int apply(int dummy) {
    return nextInt();
  }

  /**
   * Returns a deep copy of the receiver; the copy will produce identical sequences. After this call has returned, the
   * copy and the receiver have equal but separate state.
   *
   * @return a copy of the receiver.
   */
  @Override
  public Object clone() {
    AbstractDistribution copy = (AbstractDistribution) super.clone();
    if (this.randomGenerator != null) {
      copy.randomGenerator = (RandomEngine) this.randomGenerator.clone();
    }
    return copy;
  }

  /** Returns the used uniform random number generator; */
  protected RandomEngine getRandomGenerator() {
    return randomGenerator;
  }

  /**
   * Constructs and returns a new uniform random number generation engine seeded with the current time. Currently this
   * is {@link org.apache.mahout.math.jet.random.engine.MersenneTwister}.
   */
  public static RandomEngine makeDefaultGenerator() {
    return org.apache.mahout.math.jet.random.engine.RandomEngine.makeDefault();
  }

  /** Returns a random number from the distribution. */
  public abstract double nextDouble();

  /**
   * Returns a random number from the distribution; returns <tt>(int) Math.round(nextDouble())</tt>. Override this
   * method if necessary.
   */
  public int nextInt() {
    return (int) Math.round(nextDouble());
  }

  /** Sets the uniform random generator internally used. */
  protected void setRandomGenerator(RandomEngine randomGenerator) {
    this.randomGenerator = randomGenerator;
  }
}
