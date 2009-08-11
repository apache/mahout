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
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.transforms.PreferenceTransform;

import java.util.Collection;

/**
 * <p>Normalizes preference values for a user by converting them to
 * <a href="http://mathworld.wolfram.com/z-Score.html">"z-scores"</a>.
 * This process normalizes preference values to adjust for variation in mean and variance of a user's preferences.</p>
 *
 * <p>Imagine two users, one who tends to rate every movie he/she sees four or five stars, and another who uses the full
 * one to five star range when assigning ratings. This transform normalizes away the difference in scale used by the two
 * users so that both have a mean preference of 0.0 and a standard deviation of 1.0.</p>
 */
public final class ZScore implements PreferenceTransform {

  private final DataModel dataModel;
  private final Cache<Long, RunningAverageAndStdDev> meanAndStdevs;

  public ZScore(DataModel dataModel) {
    this.dataModel = dataModel;
    this.meanAndStdevs = new Cache<Long, RunningAverageAndStdDev>(new MeanStdevRetriever());
    refresh(null);
  }

  @Override
  public float getTransformedValue(Preference pref) throws TasteException {
    RunningAverageAndStdDev meanAndStdev = meanAndStdevs.get(pref.getUserID());
    if (meanAndStdev.getCount() > 1) {
      double stdev = meanAndStdev.getStandardDeviation();
      if (stdev > 0.0) {
        return (float) ((pref.getValue() - meanAndStdev.getAverage()) / stdev);
      }
    }
    return 0.0f;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    meanAndStdevs.clear();
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, dataModel);
  }

  @Override
  public String toString() {
    return "ZScore";
  }

  private class MeanStdevRetriever implements Retriever<Long, RunningAverageAndStdDev> {

    @Override
    public RunningAverageAndStdDev get(Long userID) throws TasteException {
      RunningAverageAndStdDev running = new FullRunningAverageAndStdDev();
      PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
      int size = prefs.length();
      for (int i = 0; i < size; i++) {
        running.addDatum(prefs.getValue(i));
      }
      return running;
    }
  }

}
