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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.transforms.PreferenceTransform;

import java.util.Collection;

/**
 * <p>Normalizes preference values for a {@link User} by converting them to
 * <a href="http://mathworld.wolfram.com/z-Score.html">"z-scores"</a>. This process
 * normalizes preference values to adjust for variation in mean and variance of a
 * user's preferences.</p>
 *
 * <p>Imagine two users, one who tends to rate every movie he/she sees four or five stars,
 * and another who uses the full one to five star range when assigning ratings. This
 * transform normalizes away the difference in scale used by the two users so that both
 * have a mean preference of 0.0 and a standard deviation of 1.0.</p>
 */
public final class ZScore implements PreferenceTransform {

  private final Cache<User, RunningAverageAndStdDev> meanAndStdevs;

  public ZScore() {
    this.meanAndStdevs = new Cache<User, RunningAverageAndStdDev>(new MeanStdevRetriever());
    refresh(null);
  }

  public double getTransformedValue(Preference pref) throws TasteException {
    RunningAverageAndStdDev meanAndStdev = meanAndStdevs.get(pref.getUser());
    if (meanAndStdev.getCount() > 1) {
      double stdev = meanAndStdev.getStandardDeviation();
      if (stdev > 0.0) {
        return (pref.getValue() - meanAndStdev.getAverage()) / stdev;
      }
    }
    return 0.0;
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

  @Override
  public String toString() {
    return "ZScore";
  }

  private static class MeanStdevRetriever implements Retriever<User, RunningAverageAndStdDev> {

    public RunningAverageAndStdDev get(User user) throws TasteException {
      RunningAverageAndStdDev running = new FullRunningAverageAndStdDev();
      Preference[] prefs = user.getPreferencesAsArray();
      for (int i = 0; i < prefs.length; i++) {
        running.addDatum(prefs[i].getValue());
      }
      return running;
    }
  }

}
