/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random.engine;

import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.IntFunction;

/**
 * Abstract base class for uniform pseudo-random number generating engines.
 * <p>
 * Most probability distributions are obtained by using a <b>uniform</b> pseudo-random number generation engine 
 * followed by a transformation to the desired distribution.
 * Thus, subclasses of this class are at the core of computational statistics, simulations, Monte Carlo methods, etc.
 * <p>
 * Subclasses produce uniformly distributed <tt>int</tt>'s and <tt>long</tt>'s in the closed intervals
 * <tt>[Integer.MIN_VALUE,Integer.MAX_VALUE]</tt> and <tt>[Long.MIN_VALUE,Long.MAX_VALUE]</tt>, respectively,
 * as well as <tt>float</tt>'s and <tt>double</tt>'s in the open unit intervals <tt>(0.0f,1.0f)</tt> and
 * <tt>(0.0,1.0)</tt>, respectively.
 * <p>
 * Subclasses need to override one single method only: <tt>nextInt()</tt>.
 * All other methods generating different data types or ranges are usually layered upon <tt>nextInt()</tt>.
 * <tt>long</tt>'s are formed by concatenating two 32 bit <tt>int</tt>'s.
 * <tt>float</tt>'s are formed by dividing the interval <tt>[0.0f,1.0f]</tt> into 2<sup>32</sup> sub intervals,
 * then randomly choosing one subinterval.
 * <tt>double</tt>'s are formed by dividing the interval <tt>[0.0,1.0]</tt> into 2<sup>64</sup> sub intervals,
 * then randomly choosing one subinterval.
 * <p>
 * Note that this implementation is <b>not synchronized</b>.
 *
 * @see MersenneTwister
 * @see java.util.Random
 */
public abstract class RandomEngine extends DoubleFunction implements IntFunction {

  /**
   * Equivalent to <tt>raw()</tt>. This has the effect that random engines can now be used as function objects,
   * returning a random number upon function evaluation.
   */
  @Override
  public double apply(double dummy) {
    return raw();
  }

  /**
   * Equivalent to <tt>nextInt()</tt>. This has the effect that random engines can now be used as function objects,
   * returning a random number upon function evaluation.
   */
  @Override
  public int apply(int dummy) {
    return nextInt();
  }

  /**
   * @return a 64 bit uniformly distributed random number in the open unit interval {@code (0.0,1.0)} (excluding
   * 0.0 and 1.0).
   */
  public double nextDouble() {
    double nextDouble;

    do {
      // -9.223372036854776E18 == (double) Long.MIN_VALUE
      // 5.421010862427522E-20 == 1 / Math.pow(2,64) == 1 / ((double) Long.MAX_VALUE - (double) Long.MIN_VALUE);
      nextDouble = (nextLong() - -9.223372036854776E18) * 5.421010862427522E-20;
    }
    // catch loss of precision of long --> double conversion
    while (!(nextDouble > 0.0 && nextDouble < 1.0));

    // --> in (0.0,1.0)
    return nextDouble;

    /*
      nextLong == Long.MAX_VALUE         --> 1.0
      nextLong == Long.MIN_VALUE         --> 0.0
      nextLong == Long.MAX_VALUE-1       --> 1.0
      nextLong == Long.MAX_VALUE-100000L --> 0.9999999999999946
      nextLong == Long.MIN_VALUE+1       --> 0.0
      nextLong == Long.MIN_VALUE-100000L --> 0.9999999999999946
      nextLong == 1L                     --> 0.5
      nextLong == -1L                    --> 0.5
      nextLong == 2L                     --> 0.5
      nextLong == -2L                    --> 0.5
      nextLong == 2L+100000L             --> 0.5000000000000054
      nextLong == -2L-100000L            --> 0.49999999999999456
    */
  }

  /**
   * @return a 32 bit uniformly distributed random number in the open unit interval {@code (0.0f, 1.0f)} (excluding
   * 0.0f and 1.0f).
   */
  public float nextFloat() {
    // catch loss of precision of double --> float conversion which could result in a value == 1.0F
    float nextFloat;
    do {
      nextFloat = (float) raw();
    }
    while (nextFloat >= 1.0f);

    // --> in [0.0f,1.0f)
    return nextFloat;
  }

  /**
   * @return a 32 bit uniformly distributed random number in the closed interval
   * <tt>[Integer.MIN_VALUE,Integer.MAX_VALUE]</tt>
   * (including <tt>Integer.MIN_VALUE</tt> and <tt>Integer.MAX_VALUE</tt>);
   */
  public abstract int nextInt();

  /**
   * @return a 64 bit uniformly distributed random number in the closed interval
   * <tt>[Long.MIN_VALUE,Long.MAX_VALUE]</tt>
   * (including <tt>Long.MIN_VALUE</tt> and <tt>Long.MAX_VALUE</tt>).
   */
  public long nextLong() {
    // concatenate two 32-bit strings into one 64-bit string
    return ((nextInt() & 0xFFFFFFFFL) << 32) | (nextInt() & 0xFFFFFFFFL);
  }

  /**
   * @return a 32 bit uniformly distributed random number in the open unit interval {@code (0.0, 1.0)} (excluding
   * 0.0 and 1.0).
   */
  public double raw() {
    int nextInt;
    do { // accept anything but zero
      nextInt = nextInt(); // in [Integer.MIN_VALUE,Integer.MAX_VALUE]-interval
    } while (nextInt == 0);

    // transform to (0.0,1.0)-interval
    // 2.3283064365386963E-10 == 1.0 / Math.pow(2,32)
    return (nextInt & 0xFFFFFFFFL) * 2.3283064365386963E-10;

    /*
      nextInt == Integer.MAX_VALUE   --> 0.49999999976716936
      nextInt == Integer.MIN_VALUE   --> 0.5
      nextInt == Integer.MAX_VALUE-1 --> 0.4999999995343387
      nextInt == Integer.MIN_VALUE+1 --> 0.5000000002328306
      nextInt == 1                   --> 2.3283064365386963E-10
      nextInt == -1                  --> 0.9999999997671694
      nextInt == 2                   --> 4.6566128730773926E-10
      nextInt == -2                  --> 0.9999999995343387
    */
  }
}
