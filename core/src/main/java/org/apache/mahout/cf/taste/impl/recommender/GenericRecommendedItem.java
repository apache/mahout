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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;

import java.io.Serializable;

/** <p>A simple implementation of {@link RecommendedItem}.</p> */
public final class GenericRecommendedItem implements RecommendedItem, Serializable {

  private final Item item;
  private final double value;

  /** @throws IllegalArgumentException if item is null or value is NaN */
  public GenericRecommendedItem(Item item, double value) {
    if (item == null) {
      throw new IllegalArgumentException("item is null");
    }
    if (Double.isNaN(value)) {
      throw new IllegalArgumentException("value is NaN");
    }
    this.item = item;
    this.value = value;
  }

  @Override
  public Item getItem() {
    return item;
  }

  @Override
  public double getValue() {
    return value;
  }

  @Override
  public String toString() {
    return "RecommendedItem[item:" + item + ", value:" + value + ']';
  }

  @Override
  public int hashCode() {
    return item.hashCode() ^ RandomUtils.hashDouble(value);
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof GenericRecommendedItem)) {
      return false;
    }
    GenericRecommendedItem other = (GenericRecommendedItem) o;
    return item.equals(other.item) && value == other.value;
  }

  /**
   * Defines a natural ordering from most-preferred item (highest value) to least-preferred.
   *
   * @return 1, -1, 0 as this value is less than, greater than or equal to the other's value
   */
  @Override
  public int compareTo(RecommendedItem other) {
    double otherValue = other.getValue();
    return value > otherValue ? -1 : value < otherValue ? 1 : 0;
  }

}
