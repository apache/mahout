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
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import java.util.Collection;

/**
 * <p>Implementations of this interface compute an inferred preference for a {@link User} and an {@link Item}
 * that the user has not expressed any preference for. This might be an average of other preferences scores
 * from that user, for example. This technique is sometimes called "default voting".</p>
 */
public final class AveragingPreferenceInferrer implements PreferenceInferrer {

  private static final Retriever<User, Double> RETRIEVER = new PrefRetriever();
  private static final Double ZERO = new Double(0.0);

  private final Cache<User, Double> averagePreferenceValue;

  public AveragingPreferenceInferrer(DataModel dataModel) throws TasteException {
    averagePreferenceValue = new Cache<User, Double>(RETRIEVER, dataModel.getNumUsers());
    refresh(null);
  }

  public double inferPreference(User user, Item item) throws TasteException {
    if (user == null || item == null) {
      throw new IllegalArgumentException("user or item is null");
    }
    return averagePreferenceValue.get(user);
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    averagePreferenceValue.clear();
  }

  private static final class PrefRetriever implements Retriever<User, Double> {

    public Double get(User key) {
      RunningAverage average = new FullRunningAverage();
      Preference[] prefs = key.getPreferencesAsArray();
      if (prefs.length == 0) {
        return ZERO;
      }
      for (int i = 0; i < prefs.length; i++) {
        average.addDatum(prefs[i].getValue());
      }
      return average.getAverage();
    }
  }

  @Override
  public String toString() {
    return "AveragingPreferenceInferrer";
  }

}
