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

import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.correlation.ItemCorrelation;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;

import java.util.Collection;

/**
 * See <a href="http://citeseer.ist.psu.edu/29096.html">http://citeseer.ist.psu.edu/29096.html</a>.
 */
public final class LogLikelihoodCorrelation implements ItemCorrelation {

  private final DataModel dataModel;

  public LogLikelihoodCorrelation(DataModel dataModel) {
    this.dataModel = dataModel;
  }

  public final double itemCorrelation(Item item1, Item item2) throws TasteException {
    if (item1 == null || item2 == null) {
      throw new IllegalArgumentException("item1 or item2 is null");
    }
    int preferring1and2 = dataModel.getNumUsersWithPreferenceFor(item1.getID(), item2.getID());
    int preferring1 = dataModel.getNumUsersWithPreferenceFor(item1.getID());
    int preferring2 = dataModel.getNumUsersWithPreferenceFor(item2.getID());
    int numUsers = dataModel.getNumUsers();
    double logLikelihood =
      twoLogLambda(preferring1and2, preferring1 - preferring1and2, preferring2, numUsers - preferring2);
    return 1.0 - 1.0 / (1.0 + logLikelihood);
  }

  private static double twoLogLambda(double k1, double k2, double n1, double n2) {
    double p = (k1 + k2) / (n1 + n2);
    return 2.0 * (logL(k1 / n1, k1, n1) + logL(k2 / n2, k2, n2) - logL(p, k1, n1) - logL(p, k2, n2));
  }

  private static double logL(double p, double k, double n) {
    return k * safeLog(p) + (n - k) * safeLog(1.0 - p);
  }

  private static double safeLog(double d) {
    return d <= 0.0 ? 0 : Math.log(d);
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, dataModel);
  }

  @Override
  public final String toString() {
    return "LogLikelihoodCorrelation[dataModel:" + dataModel + ']';
  }

}