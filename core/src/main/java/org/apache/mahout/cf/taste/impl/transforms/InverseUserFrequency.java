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

package org.apache.mahout.cf.taste.impl.transforms;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.transforms.PreferenceTransform;

import java.util.Collection;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicReference;

/**
 * <p>Implements an "inverse user frequency" transformation, which boosts preference values for items for which few
 * users have expressed a preference, and reduces preference values for items for which many users have expressed a
 * preference. The idea is that these "rare" items are more useful in deciding how similar two users' tastes
 * are, and so should be emphasized in other calculatioons. This idea is mentioned in <a
 * href="ftp://ftp.research.microsoft.com/pub/tr/tr-98-12.pdf">Empirical Analysis of Predictive Algorithms for
 * Collaborative Filtering</a>.</p>
 *
 * <p>A scaling factor is computed for each item by dividing the total number of users by the number of users
 * expressing a preference for that item, and taking the log of that value. The log base of this calculation can be
 * controlled in the constructor. Intuitively, the right value for the base is equal to the average number of users who
 * express a preference for each item in your model. If each item has about 100 preferences on average, 100.0 is a good
 * log base.</p>
 */
public final class InverseUserFrequency implements PreferenceTransform {

  private final DataModel dataModel;
  private final RefreshHelper refreshHelper;
  private final double logBase;
  private final AtomicReference<FastByIDMap<Double>> iufFactors;

  /**
   * <p>Creates a {@link InverseUserFrequency} transformation. Computations use the given log base.</p>
   *
   * @param dataModel {@link DataModel} from which to calculate user frequencies
   * @param logBase   calculation logarithm base
   * @throws IllegalArgumentException if dataModel is <code>null</code> or logBase is {@link Double#NaN} or &lt;= 1.0
   */
  public InverseUserFrequency(DataModel dataModel, double logBase) throws TasteException {
    if (dataModel == null) {
      throw new IllegalArgumentException("dataModel is null");
    }
    if (Double.isNaN(logBase) || logBase <= 1.0) {
      throw new IllegalArgumentException("logBase is NaN or <= 1.0");
    }
    this.dataModel = dataModel;
    this.logBase = logBase;
    this.iufFactors = new AtomicReference<FastByIDMap<Double>>(new FastByIDMap<Double>());
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        recompute();
        return null;
      }
    });
    this.refreshHelper.addDependency(this.dataModel);
    recompute();
  }

  /** @return log base used in this object's calculations */
  public double getLogBase() {
    return logBase;
  }

  @Override
  public float getTransformedValue(Preference pref) {
    Double factor = iufFactors.get().get(pref.getItemID());
    if (factor != null) {
      return (float) (pref.getValue() * factor);
    }
    return pref.getValue();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  private synchronized void recompute() throws TasteException {
    Counters itemPreferenceCounts = new Counters();
    int numUsers = 0;
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      PreferenceArray prefs = dataModel.getPreferencesFromUser(it.nextLong());
      int size = prefs.length();
      for (int i = 0; i < size; i++) {
        itemPreferenceCounts.increment(prefs.getItemID(i));
      }
      numUsers++;
    }
    FastByIDMap<Double> newIufFactors = new FastByIDMap<Double>(itemPreferenceCounts.size());
    double logFactor = Math.log(logBase);
    for (Map.Entry<Long, int[]> entry : itemPreferenceCounts.getEntrySet()) {
      newIufFactors.put(entry.getKey(),
          Math.log((double) numUsers / (double) entry.getValue()[0]) / logFactor);
    }
    iufFactors.set(newIufFactors);
  }

  @Override
  public String toString() {
    return "InverseUserFrequency[logBase:" + logBase + ']';
  }

}
