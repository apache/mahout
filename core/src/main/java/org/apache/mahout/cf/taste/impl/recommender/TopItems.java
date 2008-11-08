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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * <p>A simple class that refactors the "find top N things" logic that is used in several places.</p>
 */
public final class TopItems {

  private TopItems() {
  }

  public static List<RecommendedItem> getTopItems(int howMany,
                                                  Iterable<Item> allItems,
                                                  Rescorer<Item> rescorer,
                                                  Estimator<Item> estimator) throws TasteException {
    if (allItems == null || estimator == null) {
      throw new IllegalArgumentException("argument is null");
    }
    Queue<RecommendedItem> topItems = new PriorityQueue<RecommendedItem>(howMany + 1, Collections.reverseOrder());
    boolean full = false;
    double lowestTopValue = Double.NEGATIVE_INFINITY;
    for (Item item : allItems) {
      if (item.isRecommendable() && (rescorer == null || !rescorer.isFiltered(item))) {
        double preference = estimator.estimate(item);
        double rescoredPref = rescorer == null ? preference : rescorer.rescore(item, preference);
        if (!Double.isNaN(rescoredPref) && (!full || rescoredPref > lowestTopValue)) {
          topItems.add(new GenericRecommendedItem(item, rescoredPref));
          if (full) {
            topItems.poll();
          } else if (topItems.size() > howMany) {
            full = true;
            topItems.poll();
          }
          lowestTopValue = topItems.peek().getValue();
        }
      }
    }
    List<RecommendedItem> result = new ArrayList<RecommendedItem>(topItems.size());
    result.addAll(topItems);
    Collections.sort(result);
    return result;
  }

  public static List<User> getTopUsers(int howMany,
                                       Iterable<? extends User> allUsers,
                                       Rescorer<User> rescorer,
                                       Estimator<User> estimator) throws TasteException {
    Queue<SimilarUser> topUsers = new PriorityQueue<SimilarUser>(howMany + 1, Collections.reverseOrder());
    boolean full = false;
    double lowestTopValue = Double.NEGATIVE_INFINITY;
    for (User user : allUsers) {
      if (rescorer != null && rescorer.isFiltered(user)) {
        continue;
      }
      double similarity = estimator.estimate(user);
      double rescoredSimilarity = rescorer == null ? similarity : rescorer.rescore(user, similarity);
      if (!Double.isNaN(rescoredSimilarity) && (!full || rescoredSimilarity > lowestTopValue)) {
        topUsers.add(new SimilarUser(user, similarity));
        if (full) {
          topUsers.poll();
        } else if (topUsers.size() > howMany) {
          full = true;
          topUsers.poll();
        }
        lowestTopValue = topUsers.peek().getSimilarity();
      }
    }
    List<SimilarUser> sorted = new ArrayList<SimilarUser>(topUsers.size());
    sorted.addAll(topUsers);
    Collections.sort(sorted);
    List<User> result = new ArrayList<User>(sorted.size());
    for (SimilarUser similarUser : sorted) {
      result.add(similarUser.getUser());
    }
    return result;
  }

  /**
   * <p>Thanks to tsmorton for suggesting this functionality and writing part of the code.</p>
   *
   * @see GenericItemSimilarity#GenericItemSimilarity(Iterable, int)
   * @see GenericItemSimilarity#GenericItemSimilarity(org.apache.mahout.cf.taste.similarity.ItemSimilarity,
   *  org.apache.mahout.cf.taste.model.DataModel, int)
   */
  public static List<GenericItemSimilarity.ItemItemSimilarity> getTopItemItemSimilarities(
          int howMany, Iterable<GenericItemSimilarity.ItemItemSimilarity> allSimilarities) {
    Queue<GenericItemSimilarity.ItemItemSimilarity> topSimilarities =
            new PriorityQueue<GenericItemSimilarity.ItemItemSimilarity>(howMany + 1, Collections.reverseOrder());
    boolean full = false;
    double lowestTopValue = Double.NEGATIVE_INFINITY;
    for (GenericItemSimilarity.ItemItemSimilarity similarity : allSimilarities) {
      double value = similarity.getValue();
      if (!full || value > lowestTopValue) {
        topSimilarities.add(similarity);
        if (full) {
          topSimilarities.poll();
        } else if (topSimilarities.size() > howMany) {
          full = true;
          topSimilarities.poll();
        }
        lowestTopValue = topSimilarities.peek().getValue();
      }
    }
    List<GenericItemSimilarity.ItemItemSimilarity> result =
      new ArrayList<GenericItemSimilarity.ItemItemSimilarity>(topSimilarities.size());
    result.addAll(topSimilarities);
    Collections.sort(result);
    return result;
  }

  public static interface Estimator<T> {

    double estimate(T thing) throws TasteException;
  }

}
