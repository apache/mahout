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
 * u

/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random.engine;

import java.util.Date;
/**
 * Quick medium quality uniform pseudo-random number generator.
 *
 * Produces uniformly distributed <tt>int</tt>'s and <tt>long</tt>'s in the closed intervals <tt>[Integer.MIN_VALUE,Integer.MAX_VALUE]</tt> and <tt>[Long.MIN_VALUE,Long.MAX_VALUE]</tt>, respectively, 
 * as well as <tt>float</tt>'s and <tt>double</tt>'s in the open unit intervals <tt>(0.0f,1.0f)</tt> and <tt>(0.0,1.0)</tt>, respectively.
 * <p>
 * The seed can be any integer satisfying <tt>0 &lt; 4*seed+1 &lt; 2<sup>32</sup></tt>.
 * In other words, there must hold <tt>seed &gt;= 0 && seed &lt; 1073741823</tt>.
 * <p>
 * <b>Quality:</b> This generator follows the multiplicative congruential method of the form                    
 * <dt>
 * <tt>z(i+1) = a * z(i) (mod m)</tt> with
 * <tt>a=663608941 (=0X278DDE6DL), m=2<sup>32</sup></tt>.
 * <dt>
 * <tt>z(i)</tt> assumes all different values <tt>0 &lt; 4*seed+1 &lt; m</tt> during a full period of 2<sup>30</sup>.
 *
 * <p>
 * <b>Performance:</b> TO_DO
 * <p>
 * <b>Implementation:</b> TO_DO
 * <p>
 * Note that this implementation is <b>not synchronized</b>.                                  
 * <p>
 *
 * @see MersenneTwister
 * @see java.util.Random
 */
public class DRand extends RandomEngine {

  private int current;
  private static final int DEFAULT_SEED = 1;

  /** Constructs and returns a random number generator with a default seed, which is a <b>constant</b>. */
  public DRand() {
    this(DEFAULT_SEED);
  }

  /**
   * Constructs and returns a random number generator with the given seed.
   *
   * @param seed should not be 0, in such a case <tt>DRand.DEFAULT_SEED</tt> is substituted.
   */
  public DRand(int seed) {
    setSeed(seed);
  }

  /**
   * Constructs and returns a random number generator seeded with the given date.
   *
   * @param d typically <tt>new Date()</tt>
   */
  public DRand(Date d) {
    this((int) d.getTime());
  }

  /**
   * Returns a 32 bit uniformly distributed random number in the closed interval <tt>[Integer.MIN_VALUE,Integer.MAX_VALUE]</tt>
   * (including <tt>Integer.MIN_VALUE</tt> and <tt>Integer.MAX_VALUE</tt>).
   */
  @Override
  public int nextInt() {
    current *= 0x278DDE6D;     /* z(i+1)=a*z(i) (mod 2**32) */
    // a == 0x278DDE6D == 663608941

    return current;
  }

  /**
   * Sets the receiver's seed. This method resets the receiver's entire internal state. The following condition must
   * hold: <tt>seed &gt;= 0 && seed &lt; (2<sup>32</sup>-1) / 4</tt>.
   *
   * @param seed if the above condition does not hold, a modified seed that meets the condition is silently
   *             substituted.
   */
  protected void setSeed(int seed) {
    if (seed < 0) {
      seed = -seed;
    }
    int limit = (int) ((Math.pow(2, 32) - 1) / 4); // --> 536870911
    if (seed >= limit) {
      seed >>= 3;
    }

    this.current = 4 * seed + 1;
  }
}
