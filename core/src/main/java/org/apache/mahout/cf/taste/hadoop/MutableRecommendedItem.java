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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.RandomUtils;

/**
 * Mutable variant of {@link RecommendedItem}
 */
public class MutableRecommendedItem implements RecommendedItem {

  private long itemID;
  private float value;

  public MutableRecommendedItem() {}

  public MutableRecommendedItem(long itemID, float value) {
    this.itemID = itemID;
    this.value = value;
  }

  @Override
  public long getItemID() {
    return itemID;
  }

  @Override
  public float getValue() {
    return value;
  }

  public void setItemID(long itemID) {
    this.itemID = itemID;
  }

  public void set(long itemID, float value) {
    this.itemID = itemID;
    this.value = value;
  }

  public void capToMaxValue(float maxValue) {
    if (value > maxValue) {
      value = maxValue;
    }
  }

  @Override
  public String toString() {
    return "MutableRecommendedItem[item:" + itemID + ", value:" + value + ']';
  }

  @Override
  public int hashCode() {
    return (int) itemID ^ RandomUtils.hashFloat(value);
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof MutableRecommendedItem)) {
      return false;
    }
    RecommendedItem other = (RecommendedItem) o;
    return itemID == other.getItemID() && value == other.getValue();
  }
}
