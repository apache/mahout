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
 * <p>A {@link Preference} encapsulates an item and a preference value, which indicates the strength of the
 * preference for it. {@link Preference}s are associated to users.</p>
 */
public interface Preference {

  /** @return ID of user who prefers the item */
  long getUserID();

  /** @return item ID that is preferred */
  long getItemID();

  /**
   * @return strength of the preference for that item. Zero should indicate "no preference either way"; positive values
   *         indicate preference and negative values indicate dislike
   */
  float getValue();

  /**
   * Sets the strength of the preference for this item
   *
   * @param value new preference
   */
  void setValue(float value);

}
