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

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.GenericUserSimilarity;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;

/**
 * <p>
 * A simple class that refactors the "find top N things" logic that is used in several places.
 * </p>
 */
public final class TopItems {
  
  private static final long[] NO_IDS = new long[0];
  
  private TopItems() { }
  
  public static List<RecommendedItem> getTopItems(int howMany,
                                                  LongPrimitiveIterator possibleItemIDs,
                                                  IDRescorer rescorer,
                                                  Estimator<Long> estimator) throws TasteException {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1317
    Preconditions.checkArgument(possibleItemIDs != null, "possibleItemIDs is null");
    Preconditions.checkArgument(estimator != null, "estimator is null");

//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
    Queue<RecommendedItem> topItems = new PriorityQueue<>(howMany + 1,
      Collections.reverseOrder(ByValueRecommendedItemComparator.getInstance()));
    boolean full = false;
    double lowestTopValue = Double.NEGATIVE_INFINITY;
    while (possibleItemIDs.hasNext()) {
      long itemID = possibleItemIDs.next();
      if (rescorer == null || !rescorer.isFiltered(itemID)) {
        double preference;
//IC see: https://issues.apache.org/jira/browse/MAHOUT-247
        try {
          preference = estimator.estimate(itemID);
        } catch (NoSuchItemException nsie) {
          continue;
        }
        double rescoredPref = rescorer == null ? preference : rescorer.rescore(itemID, preference);
        if (!Double.isNaN(rescoredPref) && (!full || rescoredPref > lowestTopValue)) {
          topItems.add(new GenericRecommendedItem(itemID, (float) rescoredPref));
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
    int size = topItems.size();
    if (size == 0) {
      return Collections.emptyList();
    }
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
    List<RecommendedItem> result = new ArrayList<>(size);
    result.addAll(topItems);
//IC see: https://issues.apache.org/jira/browse/MAHOUT-370
    Collections.sort(result, ByValueRecommendedItemComparator.getInstance());
    return result;
  }
  
  public static long[] getTopUsers(int howMany,
//IC see: https://issues.apache.org/jira/browse/MAHOUT-158
                                   LongPrimitiveIterator allUserIDs,
                                   IDRescorer rescorer,
                                   Estimator<Long> estimator) throws TasteException {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
    Queue<SimilarUser> topUsers = new PriorityQueue<>(howMany + 1, Collections.reverseOrder());
    boolean full = false;
    double lowestTopValue = Double.NEGATIVE_INFINITY;
    while (allUserIDs.hasNext()) {
      long userID = allUserIDs.next();
      if (rescorer != null && rescorer.isFiltered(userID)) {
        continue;
      }
      double similarity;
//IC see: https://issues.apache.org/jira/browse/MAHOUT-247
      try {
        similarity = estimator.estimate(userID);
      } catch (NoSuchUserException nsue) {
        continue;
      }
      double rescoredSimilarity = rescorer == null ? similarity : rescorer.rescore(userID, similarity);
      if (!Double.isNaN(rescoredSimilarity) && (!full || rescoredSimilarity > lowestTopValue)) {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-882
//IC see: https://issues.apache.org/jira/browse/MAHOUT-881
        topUsers.add(new SimilarUser(userID, rescoredSimilarity));
        if (full) {
          topUsers.poll();
        } else if (topUsers.size() > howMany) {
          full = true;
          topUsers.poll();
        }
        lowestTopValue = topUsers.peek().getSimilarity();
      }
    }
    int size = topUsers.size();
    if (size == 0) {
      return NO_IDS;
    }
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
    List<SimilarUser> sorted = new ArrayList<>(size);
    sorted.addAll(topUsers);
    Collections.sort(sorted);
    long[] result = new long[size];
    int i = 0;
    for (SimilarUser similarUser : sorted) {
      result[i++] = similarUser.getUserID();
    }
    return result;
  }
  
  /**
   * <p>
   * Thanks to tsmorton for suggesting this functionality and writing part of the code.
   * </p>
   * 
   * @see GenericItemSimilarity#GenericItemSimilarity(Iterable, int)
   * @see GenericItemSimilarity#GenericItemSimilarity(org.apache.mahout.cf.taste.similarity.ItemSimilarity,
   *      org.apache.mahout.cf.taste.model.DataModel, int)
   */
  public static List<GenericItemSimilarity.ItemItemSimilarity> getTopItemItemSimilarities(
    int howMany, Iterator<GenericItemSimilarity.ItemItemSimilarity> allSimilarities) {
    
    Queue<GenericItemSimilarity.ItemItemSimilarity> topSimilarities
      = new PriorityQueue<>(howMany + 1, Collections.reverseOrder());
    boolean full = false;
    double lowestTopValue = Double.NEGATIVE_INFINITY;
//IC see: https://issues.apache.org/jira/browse/MAHOUT-661
    while (allSimilarities.hasNext()) {
      GenericItemSimilarity.ItemItemSimilarity similarity = allSimilarities.next();
      double value = similarity.getValue();
      if (!Double.isNaN(value) && (!full || value > lowestTopValue)) {
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
    int size = topSimilarities.size();
    if (size == 0) {
      return Collections.emptyList();
    }
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
    List<GenericItemSimilarity.ItemItemSimilarity> result = new ArrayList<>(size);
    result.addAll(topSimilarities);
    Collections.sort(result);
    return result;
  }
  
  public static List<GenericUserSimilarity.UserUserSimilarity> getTopUserUserSimilarities(
    int howMany, Iterator<GenericUserSimilarity.UserUserSimilarity> allSimilarities) {
    
    Queue<GenericUserSimilarity.UserUserSimilarity> topSimilarities
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
      = new PriorityQueue<>(howMany + 1, Collections.reverseOrder());
    boolean full = false;
    double lowestTopValue = Double.NEGATIVE_INFINITY;
//IC see: https://issues.apache.org/jira/browse/MAHOUT-661
    while (allSimilarities.hasNext()) {
      GenericUserSimilarity.UserUserSimilarity similarity = allSimilarities.next();
      double value = similarity.getValue();
      if (!Double.isNaN(value) && (!full || value > lowestTopValue)) {
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
    int size = topSimilarities.size();
    if (size == 0) {
      return Collections.emptyList();
    }
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
    List<GenericUserSimilarity.UserUserSimilarity> result = new ArrayList<>(size);
    result.addAll(topSimilarities);
    Collections.sort(result);
    return result;
  }
  
  public interface Estimator<T> {
    double estimate(T thing) throws TasteException;
  }
  
}
