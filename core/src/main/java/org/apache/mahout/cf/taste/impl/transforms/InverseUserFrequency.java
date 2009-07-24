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
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.transforms.PreferenceTransform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
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

  private static final Logger log = LoggerFactory.getLogger(InverseUserFrequency.class);

  private final DataModel dataModel;
  private final double logBase;
  private final AtomicReference<Map<Comparable<?>, Double>> iufFactors;

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
    this.iufFactors = new AtomicReference<Map<Comparable<?>, Double>>(new FastMap<Comparable<?>, Double>());
    recompute();
  }

  /** @return log base used in this object's calculations */
  public double getLogBase() {
    return logBase;
  }

  @Override
  public double getTransformedValue(Preference pref) {
    Double factor = iufFactors.get().get(pref.getItemID());
    if (factor != null) {
      return pref.getValue() * factor;
    }
    return pref.getValue();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    try {
      recompute();
    } catch (TasteException te) {
      log.warn("Unable to refresh", te);
    }
  }

  private synchronized void recompute() throws TasteException {
    Counters<Comparable<?>> itemPreferenceCounts = new Counters<Comparable<?>>();
    int numUsers = 0;
    for (User user : dataModel.getUsers()) {
      Preference[] prefs = user.getPreferencesAsArray();
      for (Preference pref : prefs) {
        itemPreferenceCounts.increment(pref.getItemID());
      }
      numUsers++;
    }
    Map<Comparable<?>, Double> newIufFactors = new FastMap<Comparable<?>, Double>(itemPreferenceCounts.size());
    double logFactor = Math.log(logBase);
    for (Map.Entry<Comparable<?>, int[]> entry : itemPreferenceCounts.getEntrySet()) {
      newIufFactors.put(entry.getKey(),
          Math.log((double) numUsers / (double) entry.getValue()[0]) / logFactor);
    }
    iufFactors.set(Collections.unmodifiableMap(newIufFactors));
  }

  @Override
  public String toString() {
    return "InverseUserFrequency[logBase:" + logBase + ']';
  }

}
