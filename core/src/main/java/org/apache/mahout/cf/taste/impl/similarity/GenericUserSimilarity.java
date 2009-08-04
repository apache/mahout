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

package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.IteratorIterable;
import org.apache.mahout.cf.taste.impl.common.IteratorUtils;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

public final class GenericUserSimilarity implements UserSimilarity {

  private final Map<Comparable<?>, Map<Comparable<?>, Double>> similarityMaps =
          new FastMap<Comparable<?>, Map<Comparable<?>, Double>>();

  public GenericUserSimilarity(Iterable<UserUserSimilarity> similarities) {
    initSimilarityMaps(similarities);
  }

  public GenericUserSimilarity(Iterable<UserUserSimilarity> similarities, int maxToKeep) {
    Iterable<UserUserSimilarity> keptSimilarities = TopItems.getTopUserUserSimilarities(maxToKeep, similarities);
    initSimilarityMaps(keptSimilarities);
  }

  public GenericUserSimilarity(UserSimilarity otherSimilarity, DataModel dataModel) throws TasteException {
    List<? extends Comparable<?>> userIDs = IteratorUtils.iterableToList(dataModel.getUserIDs());
    Iterator<UserUserSimilarity> it = new DataModelSimilaritiesIterator(otherSimilarity, userIDs);
    initSimilarityMaps(new IteratorIterable<UserUserSimilarity>(it));
  }

  public GenericUserSimilarity(UserSimilarity otherSimilarity, DataModel dataModel, int maxToKeep)
      throws TasteException {
    List<? extends Comparable<?>> userIDs = IteratorUtils.iterableToList(dataModel.getUserIDs());
    Iterator<UserUserSimilarity> it = new DataModelSimilaritiesIterator(otherSimilarity, userIDs);
    Iterable<UserUserSimilarity> keptSimilarities =
        TopItems.getTopUserUserSimilarities(maxToKeep, new IteratorIterable<UserUserSimilarity>(it));
    initSimilarityMaps(keptSimilarities);
  }

  private void initSimilarityMaps(Iterable<UserUserSimilarity> similarities) {
    for (UserUserSimilarity uuc : similarities) {
      Comparable<?> similarityUser1 = uuc.getUserID1();
      Comparable<?> similarityUser2 = uuc.getUserID2();
      int compare = ((Comparable<Object>) similarityUser1).compareTo(similarityUser2);
      if (compare != 0) {
        // Order them -- first key should be the "smaller" one
        Comparable<?> user1;
        Comparable<?> user2;
        if (compare < 0) {
          user1 = similarityUser1;
          user2 = similarityUser2;
        } else {
          user1 = similarityUser2;
          user2 = similarityUser1;
        }
        Map<Comparable<?>, Double> map = similarityMaps.get(user1);
        if (map == null) {
          map = new FastMap<Comparable<?>, Double>();
          similarityMaps.put(user1, map);
        }
        map.put(user2, uuc.getValue());
      }
      // else similarity between user and itself already assumed to be 1.0
    }
  }

  @Override
  public double userSimilarity(Comparable<?> userID1, Comparable<?> userID2) {
    int compare = ((Comparable<Object>) userID1).compareTo(userID2);
    if (compare == 0) {
      return 1.0;
    }
    Comparable<?> first;
    Comparable<?> second;
    if (compare < 0) {
      first = userID1;
      second = userID2;
    } else {
      first = userID2;
      second = userID1;
    }
    Map<Comparable<?>, Double> nextMap = similarityMaps.get(first);
    if (nextMap == null) {
      return Double.NaN;
    }
    Double similarity = nextMap.get(second);
    return similarity == null ? Double.NaN : similarity;
  }

  @Override
  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // Do nothing
  }

  public static final class UserUserSimilarity implements Comparable<UserUserSimilarity> {

    private final Comparable<?> userID1;
    private final Comparable<?> userID2;
    private final double value;

    public UserUserSimilarity(Comparable<?> userID1, Comparable<?> userID2, double value) {
      if (userID1 == null || userID2 == null) {
        throw new IllegalArgumentException("A user is null");
      }
      if (Double.isNaN(value) || value < -1.0 || value > 1.0) {
        throw new IllegalArgumentException("Illegal value: " + value);
      }
      this.userID1 = userID1;
      this.userID2 = userID2;
      this.value = value;
    }

    public Comparable<?> getUserID1() {
      return userID1;
    }

    public Comparable<?> getUserID2() {
      return userID2;
    }

    public double getValue() {
      return value;
    }

    @Override
    public String toString() {
      return "UserUserSimilarity[" + userID1 + ',' + userID2 + ':' + value + ']';
    }

    /** Defines an ordering from highest similarity to lowest. */
    @Override
    public int compareTo(UserUserSimilarity other) {
      double otherValue = other.value;
      return value > otherValue ? -1 : value < otherValue ? 1 : 0;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof UserUserSimilarity)) {
        return false;
      }
      UserUserSimilarity otherSimilarity = (UserUserSimilarity) other;
      return otherSimilarity.userID1.equals(userID1) && otherSimilarity.userID2.equals(userID2) && otherSimilarity.value == value;
    }

    @Override
    public int hashCode() {
      return userID1.hashCode() ^ userID2.hashCode() ^ RandomUtils.hashDouble(value);
    }

  }

  private static final class DataModelSimilaritiesIterator implements Iterator<UserUserSimilarity> {

    private final UserSimilarity otherSimilarity;
    private final List<? extends Comparable<?>> userIDs;
    private final int size;
    private int i;
    private Comparable<?> userID1;
    private int j;

    private DataModelSimilaritiesIterator(UserSimilarity otherSimilarity, List<? extends Comparable<?>> userIDs) {
      this.otherSimilarity = otherSimilarity;
      this.userIDs = userIDs;
      this.size = userIDs.size();
      i = 0;
      userID1 = userIDs.get(0);
      j = 1;
    }

    @Override
    public boolean hasNext() {
      return i < size - 1;
    }

    @Override
    public UserUserSimilarity next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Comparable<?> userID2 = userIDs.get(j);
      double similarity;
      try {
        similarity = otherSimilarity.userSimilarity(userID1, userID2);
      } catch (TasteException te) {
        // ugly:
        throw new RuntimeException(te);
      }
      UserUserSimilarity result = new UserUserSimilarity(userID1, userID2, similarity);
      j++;
      if (j == size) {
        i++;
        userID1 = userIDs.get(i);
        j = i + 1;
      }
      return result;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }

  }

}