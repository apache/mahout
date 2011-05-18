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

import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;

/**
 * <p>
 * Defines ordering on {@link RecommendedItem} by the rescored value of the recommendations' estimated
 * preference value, from high to low.
 * </p>
 */
final class ByRescoreComparator implements Comparator<RecommendedItem>, Serializable {
  
  private final IDRescorer rescorer;
  
  ByRescoreComparator(IDRescorer rescorer) {
    this.rescorer = rescorer;
  }
  
  @Override
  public int compare(RecommendedItem o1, RecommendedItem o2) {
    double rescored1;
    double rescored2;
    if (rescorer == null) {
      rescored1 = o1.getValue();
      rescored2 = o2.getValue();
    } else {
      rescored1 = rescorer.rescore(o1.getItemID(), o1.getValue());
      rescored2 = rescorer.rescore(o2.getItemID(), o2.getValue());
    }
    if (rescored1 < rescored2) {
      return 1;
    } else if (rescored1 > rescored2) {
      return -1;
    } else {
      return 0;
    }
  }
  
  @Override
  public String toString() {
    return "ByRescoreComparator[rescorer:" + rescorer + ']';
  }
  
}
