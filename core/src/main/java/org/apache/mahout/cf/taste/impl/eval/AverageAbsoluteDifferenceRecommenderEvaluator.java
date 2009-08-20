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

package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * <p>A {@link org.apache.mahout.cf.taste.eval.RecommenderEvaluator} which computes the average absolute difference
 * between predicted and actual ratings for users.</p>
 *
 * <p>This algorithm is also called "mean average error".</p>
 */
public final class AverageAbsoluteDifferenceRecommenderEvaluator extends AbstractDifferenceRecommenderEvaluator {

  private static final Logger log = LoggerFactory.getLogger(AverageAbsoluteDifferenceRecommenderEvaluator.class);

  @Override
  double getEvaluation(FastByIDMap<PreferenceArray> testUserPrefs, Recommender recommender) throws TasteException {
    RunningAverage average = new FullRunningAverage();
    int count = 0;
    for (Map.Entry<Long, PreferenceArray> entry : testUserPrefs.entrySet()) {
      for (Preference realPref : entry.getValue()) {
        long testUserID = entry.getKey();
        try {
          float estimatedPreference =
              recommender.estimatePreference(testUserID, realPref.getItemID());
          if (!Float.isNaN(estimatedPreference)) {
            average.addDatum(Math.abs(realPref.getValue() - estimatedPreference));
          }
        } catch (NoSuchUserException nsue) {
          // It's possible that an item exists in the test data but not training data in which case
          // NSEE will be thrown. Just ignore it and move on.
          log.info("User exists in test data but not training data: {}", testUserID);
        } catch (NoSuchItemException nsie) {
          log.info("Item exists in test data but not training data: {}", realPref.getItemID());
        }
      }
      if (++count % 100 == 0) {
        log.info("Finished evaluation for {} prefs", count);
      }
    }
    return average.getAverage();
  }

  @Override
  public String toString() {
    return "AverageAbsoluteDifferenceRecommenderEvaluator";
  }

}
