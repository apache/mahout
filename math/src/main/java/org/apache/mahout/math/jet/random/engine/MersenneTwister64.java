/**
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
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
is hereby granted without fee, provided that the above copyright notice appear in all copies and
that both that copyright notice and this permission notice appear in supporting documentation.
CERN makes no representations about the suitability of this software for any purpose.
It is provided "as is" without expressed or implied warranty.
*/

package org.apache.mahout.math.jet.random.engine;

import java.util.Date;
/**
 * Same as <tt>MersenneTwister</tt> except that method <tt>raw()</tt> returns 64 bit random numbers instead of 32 bit random numbers.
 *
 * @see MersenneTwister
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class MersenneTwister64 extends MersenneTwister {

  /** Constructs and returns a random number generator with a default seed, which is a <b>constant</b>. */
  public MersenneTwister64() {
    super();
  }

  /**
   * Constructs and returns a random number generator with the given seed.
   *
   * @param seed should not be 0, in such a case <tt>MersenneTwister64.DEFAULT_SEED</tt> is silently substituted.
   */
  public MersenneTwister64(int seed) {
    super(seed);
  }

  /**
   * Constructs and returns a random number generator seeded with the given date.
   *
   * @param d typically <tt>new Date()</tt>
   */
  public MersenneTwister64(Date d) {
    super(d);
  }

  /**
   * Returns a 64 bit uniformly distributed random number in the open unit interval <code>(0.0,1.0)</code> (excluding
   * 0.0 and 1.0).
   */
  @Override
  public double raw() {
    return nextDouble();
  }
}
