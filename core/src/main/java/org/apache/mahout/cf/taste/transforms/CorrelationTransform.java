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

package org.apache.mahout.cf.taste.transforms;

import org.apache.mahout.cf.taste.common.Refreshable;

/**
 * <p>Implementations encapsulate some transformation on correlation values between two
 * things, where things might be {@link org.apache.mahout.cf.taste.model.User}s or {@link org.apache.mahout.cf.taste.model.Item}s or
 * something else.</p>
 */
public interface CorrelationTransform<T> extends Refreshable {

  /**
   * @param thing1
   * @param thing2
   * @param value original correlation between thing1 and thing2
   * (should be in [-1,1])
   * @return transformed correlation (should be in [-1,1])
   */
  double transformCorrelation(T thing1, T thing2, double value);

}
