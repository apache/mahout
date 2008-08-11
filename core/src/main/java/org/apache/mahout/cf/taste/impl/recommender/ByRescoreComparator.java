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

import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;

import java.util.Comparator;
import java.io.Serializable;

/**
 * <p>A simple {@link org.apache.mahout.cf.taste.recommender.Rescorer} which always returns the original score.</p>
 */
final class ByRescoreComparator implements Comparator<RecommendedItem>, Serializable {

  private final Rescorer<Item> rescorer;

  ByRescoreComparator(Rescorer<Item> rescorer) {
    if (rescorer == null) {
      throw new IllegalArgumentException("rescorer is null");
    }
    this.rescorer = rescorer;
  }

  public int compare(RecommendedItem o1, RecommendedItem o2) {
    double rescored1 = rescorer.rescore(o1.getItem(), o1.getValue());
    double rescored2 = rescorer.rescore(o2.getItem(), o2.getValue());
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
