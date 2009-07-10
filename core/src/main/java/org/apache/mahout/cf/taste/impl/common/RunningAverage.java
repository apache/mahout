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

package org.apache.mahout.cf.taste.impl.common;

/**
 * <p>Interface for classes that can keep track of a running average of a series of numbers. One can add to or remove
 * from the series, as well as update a datum in the series. The class does not actually keep track of the series of
 * values, just its running average, so it doesn't even matter if you remove/change a value that wasn't added.</p>
 */
public interface RunningAverage {

  /**
   * @param datum new item to add to the running average
   * @throws IllegalArgumentException if datum is {@link Double#NaN}
   */
  void addDatum(double datum);

  /**
   * @param datum item to remove to the running average
   * @throws IllegalArgumentException if datum is {@link Double#NaN}
   * @throws IllegalStateException    if count is 0
   */
  void removeDatum(double datum);

  /**
   * @param delta amount by which to change a datum in the running average
   * @throws IllegalArgumentException if delta is {@link Double#NaN}
   * @throws IllegalStateException    if count is 0
   */
  void changeDatum(double delta);

  int getCount();

  double getAverage();

}
