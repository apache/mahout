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

import org.apache.mahout.math.PersistentObject;

/**
 * Deterministic seed generator for pseudo-random number generators.
 * The sole purpose of this class is to decorrelate seeds and uniform random number generators.
 * (If a generator would be used to generate seeds for itself, the result could be correlations.)
 * <p>
 * This class has entirelly deterministic behaviour:
 * Constructing two instances with the same parameters at any two distinct points in time will produce identical seed sequences.
 * However, it does not (at all) generate uniformly distributed numbers. Do not use it as a uniformly distributed random number generator! 
 * <p>
 * Each generated sequence of seeds has a period of 10<sup>9</sup> numbers.
 * Internally uses {@link RandomSeedTable}.
 *
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class RandomSeedGenerator extends PersistentObject {

  private int row;
  private final int column;

  /** Constructs and returns a new seed generator. */
  public RandomSeedGenerator() {
    this(0, 0);
  }

  /**
   * Constructs and returns a new seed generator; you normally won't need to use this method. <p> The position
   * <tt>[row,column]</tt> indicates the iteration starting point within a (virtual) seed matrix. The seed matrix is a
   * n*m matrix with <tt>1 + Integer.MAX_VALUE</tt> (virtual) rows and <tt>RandomSeedTable.COLUMNS</tt> columns.
   * Successive calls to method <tt>nextSeed()</tt> will cycle over the given column, in ascending order:
   * <tt>nextSeed()</tt> returns the seed <tt>s[row,column], s[row+1,column], ... s[Integer.MAX_VALUE,column],
   * s[0,column], s[1,column], ...</tt>
   *
   * @param row    should be in <tt>[0,Integer.MAX_VALUE]</tt>.
   * @param column should be in <tt>[0,RandomSeedTable.COLUMNS - 1]</tt>.
   */
  public RandomSeedGenerator(int row, int column) {
    this.row = row;
    this.column = column;
  }


  /** Returns the next seed. */
  public int nextSeed() {
    return RandomSeedTable.getSeedAtRowColumn(row++, column);
  }

}
