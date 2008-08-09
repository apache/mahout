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

package org.apache.mahout.cf.taste.impl.correlation;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.correlation.PreferenceInferrer;
import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.impl.model.ByItemPreferenceComparator;
import org.apache.mahout.cf.taste.impl.model.ByValuePreferenceComparator;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.locks.ReentrantLock;

/**
 * <p>Like {@link PearsonCorrelation}, but compares relative ranking of preference values instead of preference
 * values themselves. That is, each {@link User}'s preferences are sorted and then assign a rank as their preference
 * value, with 1 being assigned to the least preferred item. Then the Pearson itemCorrelation of these rank values is
 * computed.</p>
 */
public final class SpearmanCorrelation implements UserCorrelation {

  private final UserCorrelation rankingUserCorrelation;

  public SpearmanCorrelation(DataModel dataModel) throws TasteException {
    if (dataModel == null) {
      throw new IllegalArgumentException("dataModel is null");
    }
    this.rankingUserCorrelation = new PearsonCorrelation(dataModel);
  }

  public SpearmanCorrelation(UserCorrelation rankingUserCorrelation) {
    if (rankingUserCorrelation == null) {
      throw new IllegalArgumentException("rankingUserCorrelation is null");
    }
    this.rankingUserCorrelation = rankingUserCorrelation;
  }

  public double userCorrelation(User user1, User user2) throws TasteException {
    if (user1 == null || user2 == null) {
      throw new IllegalArgumentException("user1 or user2 is null");
    }
    return rankingUserCorrelation.userCorrelation(new RankedPreferenceUser(user1),
                                                  new RankedPreferenceUser(user2));
  }

  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    rankingUserCorrelation.setPreferenceInferrer(inferrer);
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, rankingUserCorrelation);
  }


  /**
   * <p>A simple {@link User} decorator which will always return the underlying {@link User}'s
   * preferences in order by value.</p>
   */
  private static final class RankedPreferenceUser implements User {

    private final User delegate;

    private RankedPreferenceUser(User delegate) {
      this.delegate = delegate;
    }

    public Object getID() {
      return delegate.getID();
    }

    public Preference getPreferenceFor(Object itemID) {
      throw new UnsupportedOperationException();
    }

    public Iterable<Preference> getPreferences() {
      return Arrays.asList(getPreferencesAsArray());
    }

    public Preference[] getPreferencesAsArray() {
      Preference[] source = delegate.getPreferencesAsArray();
      int length = source.length;
      Preference[] sortedPrefs = new Preference[length];
      System.arraycopy(source, 0, sortedPrefs, 0, length);
      Arrays.sort(sortedPrefs, ByValuePreferenceComparator.getInstance());
      for (int i = 0; i < length; i++) {
        sortedPrefs[i] = new GenericPreference(this, sortedPrefs[i].getItem(), (double) (i + 1));
      }
      Arrays.sort(sortedPrefs, ByItemPreferenceComparator.getInstance());
      return sortedPrefs;
    }

    @Override
    public int hashCode() {
      return delegate.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof RankedPreferenceUser && delegate.equals(((RankedPreferenceUser) o).delegate);
    }

    public int compareTo(User user) {
      return delegate.compareTo(user);
    }

    @Override
    public String toString() {
      return "RankedPreferenceUser[user:" + delegate + ']';
    }

  }

}
