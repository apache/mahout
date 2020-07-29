/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import java.util.Random;

import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.IntFunction;

public abstract class AbstractDistribution extends DoubleFunction implements IntFunction {

  private Random randomGenerator;

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected AbstractDistribution() {
  }
  
  protected Random getRandomGenerator() {
    return randomGenerator;
  }

  protected double randomDouble() {
    return randomGenerator.nextDouble();
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
   * Returns a random number from the distribution.
   * @return A new sample from this distribution.
   */
  public abstract double nextDouble();

  /**
   * @return
   * A random number from the distribution; returns <tt>(int) Math.round(nextDouble())</tt>. Override this
   * method if necessary.
   */
  public abstract int nextInt();

  /**
   * Sets the uniform random generator internally used.
   * @param randomGenerator the new PRNG
   */
  public void setRandomGenerator(Random randomGenerator) {
    this.randomGenerator = randomGenerator;
  }
}
