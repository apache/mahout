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

package org.apache.mahout.cf.taste.impl.model;

import java.io.Serializable;

import org.apache.mahout.cf.taste.model.Preference;

/**
 * Encapsulates a simple boolean "preference" for an item whose value does not matter (is fixed at 1.0). This
 * is appropriate in situations where users conceptually have only a general "yes" preference for items,
 * rather than a spectrum of preference values.
 */
public final class BooleanPreference implements Preference, Serializable {
  
  private final long userID;
  private final long itemID;
  
  public BooleanPreference(long userID, long itemID) {
    this.userID = userID;
    this.itemID = itemID;
  }
  
  @Override
  public long getUserID() {
    return userID;
  }
  
  @Override
  public long getItemID() {
    return itemID;
  }
  
  @Override
  public float getValue() {
    return 1.0f;
  }
  
  @Override
  public void setValue(float value) {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public String toString() {
    return "BooleanPreference[userID: " + userID + ", itemID:" + itemID + ']';
  }
  
}
