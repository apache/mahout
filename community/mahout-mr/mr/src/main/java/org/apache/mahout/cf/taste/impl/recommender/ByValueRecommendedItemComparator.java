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

import java.io.Serializable;
import java.util.Comparator;

import org.apache.mahout.cf.taste.recommender.RecommendedItem;

/**
 * Defines a natural ordering from most-preferred item (highest value) to least-preferred.
 */
public final class ByValueRecommendedItemComparator implements Comparator<RecommendedItem>, Serializable {

  private static final Comparator<RecommendedItem> INSTANCE = new ByValueRecommendedItemComparator();

  public static Comparator<RecommendedItem> getInstance() {
    return INSTANCE;
  }

  @Override
  public int compare(RecommendedItem o1, RecommendedItem o2) {
    float value1 = o1.getValue();
    float value2 = o2.getValue();
    return value1 > value2 ? -1 : value1 < value2 ? 1 : 0;
  }

}
