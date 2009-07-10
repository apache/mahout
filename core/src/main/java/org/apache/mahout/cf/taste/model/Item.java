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

package org.apache.mahout.cf.taste.model;

/**
 * <p>Implementations of this interface represent items that {@link User}s have preferences for, and which can be
 * recommended to them. {@link Item}s must have a unique ID of some kind, and must be {@link Comparable}.</p>
 */
public interface Item extends Comparable<Item> {

  /** @return unique ID for this item */
  Object getID();

  /**
   * @return true if and only if this {@link Item} can be recommended to a user; for example, this could be false for an
   *         {@link Item} that is no longer available but which remains valuable for recommendation
   */
  boolean isRecommendable();

}
