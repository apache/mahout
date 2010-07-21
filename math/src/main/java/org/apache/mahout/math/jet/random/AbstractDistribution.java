/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.math.function.IntFunction;
import org.apache.mahout.math.jet.random.engine.RandomEngine;

public abstract class AbstractDistribution extends PersistentObject implements UnaryFunction, IntFunction {

  protected RandomEngine randomGenerator;

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected AbstractDistribution() {
  }

  /**
   * Equivalent to <tt>nextDouble()</tt>. This has the effect that distributions can now be used as function objects,
   * returning a random number upon function evaluation.
   */
  public double apply(double dummy) {
    return nextDouble();
  }

  /**
   * Equivalent to <tt>nextInt()</tt>. This has the effect that distributions can now be used as function objects,
   * returning a random number upon function evaluation.
   */
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
  
  public byte nextByte() {
    return (byte)nextInt();
  }
  
  public char nextChar() {
    return (char)nextInt();
  }
  
  public long nextLong() {
    return Math.round(nextDouble());
  }
  
  public float nextFloat() {
    return (float)nextDouble();
  }

  /** Sets the uniform random generator internally used. */
  protected void setRandomGenerator(RandomEngine randomGenerator) {
    this.randomGenerator = randomGenerator;
  }
}
