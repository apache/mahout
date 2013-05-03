/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.math.function;

public final class TimesFunction extends DoubleDoubleFunction {

  /**
   * Computes the product of two numbers.
   *
   * @param x first argument
   * @param y second argument
   * @return the product
   */
  @Override
  public double apply(double x, double y) {
    return x * y;
  }

  /**
   * x * 0 = y only if y = 0
   * @return true iff f(x, 0) = x for any x
   */
  @Override
  public boolean isLikeRightPlus() {
    return false;
  }

  /**
   * 0 * y = 0 for any y
   * @return true iff f(0, y) = 0 for any y
   */
  @Override
  public boolean isLikeLeftMult() {
    return true;
  }

  /**
   * x * 0 = 0 for any x
   * @return true iff f(x, 0) = 0 for any x
   */
  @Override
  public boolean isLikeRightMult() {
    return true;
  }

  /**
   * x * y = y * x for any x, y
   * @return true iff f(x, y) = f(y, x) for any x, y
   */
  @Override
  public boolean isCommutative() {
    return true;
  }

  /**
   * x * (y * z) = (x * y) * z for any x, y, z
   * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
   */
  @Override
  public boolean isAssociative() {
    return true;
  }
}
