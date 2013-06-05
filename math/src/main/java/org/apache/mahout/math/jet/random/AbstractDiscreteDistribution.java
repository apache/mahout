/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

/**
 * Abstract base class for all discrete distributions.
 *
 */
public abstract class AbstractDiscreteDistribution extends AbstractDistribution {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected AbstractDiscreteDistribution() {
  }

  /** Returns a random number from the distribution; returns <tt>(double) nextInt()</tt>. */
  @Override
  public double nextDouble() {
    return nextInt();
  }

}
