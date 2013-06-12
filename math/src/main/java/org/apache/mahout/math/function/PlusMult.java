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

package org.apache.mahout.math.function;

import org.apache.mahout.math.jet.math.Constants;

/**
 * Only for performance tuning of compute intensive linear algebraic computations.
 * Constructs functions that return one of
 * <ul>
 * <li><tt>a + b*constant</tt>
 * <li><tt>a - b*constant</tt>
 * <li><tt>a + b/constant</tt>
 * <li><tt>a - b/constant</tt>
 * </ul> 
 * <tt>a</tt> and <tt>b</tt> are variables, <tt>constant</tt> is fixed, but for performance reasons publicly accessible.
 * Intended to be passed to <tt>matrix.assign(otherMatrix,function)</tt> methods.
 */

public final class PlusMult extends DoubleDoubleFunction {

  private double multiplicator;

  public PlusMult(double multiplicator) {
    this.multiplicator = multiplicator;
  }

  /** Returns the result of the function evaluation. */
  @Override
  public double apply(double a, double b) {
    return a + b * multiplicator;
  }

  /** <tt>a - b*constant</tt>. */
  public static PlusMult minusMult(double constant) {
    return new PlusMult(-constant);
  }

  /** <tt>a + b*constant</tt>. */
  public static PlusMult plusMult(double constant) {
    return new PlusMult(constant);
  }

  public double getMultiplicator() {
    return multiplicator;
  }

  /**
   * x + 0 * c = x
   * @return true iff f(x, 0) = x for any x
   */
  @Override
  public boolean isLikeRightPlus() {
    return true;
  }

  /**
   * 0 + y * c = y * c != 0
   * @return true iff f(0, y) = 0 for any y
   */
  @Override
  public boolean isLikeLeftMult() {
    return false;
  }

  /**
   * x + 0 * c = x != 0
   * @return true iff f(x, 0) = 0 for any x
   */
  @Override
  public boolean isLikeRightMult() {
    return false;
  }

  /**
   * x + y * c = y + x * c iff c = 1
   * @return true iff f(x, y) = f(y, x) for any x, y
   */
  @Override
  public boolean isCommutative() {
    return Math.abs(multiplicator - 1.0) < Constants.EPSILON;
  }

  /**
   * f(x, f(y, z)) = x + c * (y + c * z) = x + c * y + c^2  * z
   * f(f(x, y), z) = (x + c * y) + c * z = x + c * y + c * z
   * true only for c = 0 or c = 1
   * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
   */
  @Override
  public boolean isAssociative() {
    return Math.abs(multiplicator - 0.0) < Constants.EPSILON
        || Math.abs(multiplicator - 1.0) < Constants.EPSILON;
  }

  public void setMultiplicator(double multiplicator) {
    this.multiplicator = multiplicator;
  }
}
