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
import org.apache.mahout.cf.taste.impl.correlation.GenericItemCorrelation;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

/**
 * <p>A simple class that refactors the "find top N recommended items" logic that is used in
 * several places in Taste.</p>
 */
public final class TopItems {

  private TopItems() {
  }

  public static List<RecommendedItem> getTopItems(int howMany,
                                                  Iterable<Item> allItems,
                                                  Rescorer<Item> rescorer,
                                                  Estimator<Item> estimator) throws TasteException {
    if (allItems == null || rescorer == null || estimator == null) {
      throw new IllegalArgumentException("argument is null");
    }
    LinkedList<RecommendedItem> topItems = new LinkedList<RecommendedItem>();
    boolean full = false;
    for (Item item : allItems) {
      if (item.isRecommendable() && !rescorer.isFiltered(item)) {
        double preference = estimator.estimate(item);
        double rescoredPref = rescorer.rescore(item, preference);
        if (!Double.isNaN(rescoredPref) && (!full || rescoredPref > topItems.getLast().getValue())) {
          // I think this is faster than Collections.binarySearch() over a LinkedList since our
          // comparisons are cheap, which binarySearch() economizes at the expense of more traversals.
          // We also know that the right position tends to be at the end of the list.
          ListIterator<RecommendedItem> iterator = topItems.listIterator(topItems.size());
          while (iterator.hasPrevious()) {
            if (rescoredPref <= iterator.previous().getValue()) {
              iterator.next();
              break;
            }
          }
          iterator.add(new GenericRecommendedItem(item, rescoredPref));
          if (full) {
            topItems.removeLast();
          } else if (topItems.size() > howMany) {
            full = true;
            topItems.removeLast();
          }
        }
      }
    }
    return topItems;
  }

  public static List<User> getTopUsers(int howMany,
                                       Iterable<User> allUsers,
                                       Rescorer<User> rescorer,
                                       Estimator<User> estimator) throws TasteException {
    LinkedList<SimilarUser> topUsers = new LinkedList<SimilarUser>();
    boolean full = false;
    for (User user : allUsers) {
      if (rescorer.isFiltered(user)) {
        continue;
      }
      double similarity = estimator.estimate(user);
      double rescoredSimilarity = rescorer.rescore(user, similarity);
      if (!Double.isNaN(rescoredSimilarity) &&
          (!full || rescoredSimilarity > topUsers.getLast().getSimilarity())) {
        ListIterator<SimilarUser> iterator = topUsers.listIterator(topUsers.size());
        while (iterator.hasPrevious()) {
          if (rescoredSimilarity <= iterator.previous().getSimilarity()) {
            iterator.next();
            break;
          }
        }
        iterator.add(new SimilarUser(user, similarity));
        if (full) {
          topUsers.removeLast();
        } else if (topUsers.size() > howMany) {
          full = true;
          topUsers.removeLast();
        }
      }
    }
    List<User> result = new ArrayList<User>(topUsers.size());
    for (SimilarUser similarUser : topUsers) {
      result.add(similarUser.getUser());
    }
    return result;
  }

  /**
   * <p>Thanks to tsmorton for suggesting this functionality and writing part of the code.</p>
   *
   * @see GenericItemCorrelation#GenericItemCorrelation(Iterable, int)
   * @see GenericItemCorrelation#GenericItemCorrelation(org.apache.mahout.cf.taste.correlation.ItemCorrelation , org.apache.mahout.cf.taste.model.DataModel , int)
   */
  public static List<GenericItemCorrelation.ItemItemCorrelation> getTopItemItemCorrelations(
          int howMany, Iterable<GenericItemCorrelation.ItemItemCorrelation> allCorrelations) {
    LinkedList<GenericItemCorrelation.ItemItemCorrelation> topCorrelations =
            new LinkedList<GenericItemCorrelation.ItemItemCorrelation>();
    boolean full = false;
    for (GenericItemCorrelation.ItemItemCorrelation correlation : allCorrelations) {
      double value = correlation.getValue();
      if (!full || value > topCorrelations.getLast().getValue()) {
        ListIterator<GenericItemCorrelation.ItemItemCorrelation> iterator =
                topCorrelations.listIterator(topCorrelations.size());
        while (iterator.hasPrevious()) {
          if (value <= iterator.previous().getValue()) {
            iterator.next();
            break;
          }
        }
        iterator.add(correlation);
        if (full) {
          topCorrelations.removeLast();
        } else if (topCorrelations.size() > howMany) {
          full = true;
          topCorrelations.removeLast();
        }
      }
    }
    return topCorrelations;
  }

  public static interface Estimator<T> {

    double estimate(T thing) throws TasteException;
  }

  // Hmm, should this be exposed publicly like RecommendedItem?
  private static class SimilarUser implements User {

    private final User user;
    private final double similarity;

    private SimilarUser(User user, double similarity) {
      this.user = user;
      this.similarity = similarity;
    }

    public Object getID() {
      return user.getID();
    }

    public Preference getPreferenceFor(Object itemID) {
      return user.getPreferenceFor(itemID);
    }

    public Iterable<Preference> getPreferences() {
      return user.getPreferences();
    }

    public Preference[] getPreferencesAsArray() {
      return user.getPreferencesAsArray();
    }

    User getUser() {
      return user;
    }

    double getSimilarity() {
      return similarity;
    }

    @Override
    public int hashCode() {
      return user.hashCode() ^ Double.valueOf(similarity).hashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof SimilarUser)) {
        return false;
      }
      SimilarUser other = (SimilarUser) o;
      return user.equals(other.user) && similarity == other.similarity;
    }

    public int compareTo(User user) {
      return this.user.compareTo(user);
    }
  }

}
