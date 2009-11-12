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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.SamplingLongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;

/**
 * Simple helper class for running load on a Recommender.
 */
public final class LoadEvaluator {

  private static final Logger log = LoggerFactory.getLogger(LoadEvaluator.class);

  private LoadEvaluator() {
  }

  public static void runLoad(Recommender recommender) throws TasteException {

    DataModel dataModel = recommender.getDataModel();
    int numUsers = dataModel.getNumUsers();
    double sampleRate = 1000.0 / numUsers;
    LongPrimitiveIterator userSampler =
        SamplingLongPrimitiveIterator.maybeWrapIterator(dataModel.getUserIDs(), sampleRate);
    RunningAverage recommendationTime = new FullRunningAverageAndStdDev();
    recommender.recommend(userSampler.next(), 10); // Warm up
    Collection<Callable<Object>> callables = new ArrayList<Callable<Object>>();
    while (userSampler.hasNext()) {
      callables.add(new LoadCallable(recommender, userSampler.next(), recommendationTime));
    }
    AbstractDifferenceRecommenderEvaluator.execute(callables);
    logStats(recommendationTime);
  }

  private static void logStats(RunningAverage recommendationTime) {
    Runtime runtime = Runtime.getRuntime();
    System.gc();
    log.info("Average time per recommendation: " + (int) recommendationTime.getAverage() +
             "ms; approx. memory used: " + ((runtime.totalMemory() - runtime.freeMemory()) / 1000000) + "MB");

  }

  private static class LoadCallable implements Callable<Object> {

    private final Recommender recommender;
    private final long userID;
    private final RunningAverage recommendationTime;

    private LoadCallable(Recommender recommender, long userID, RunningAverage recommendationTime) {
      this.recommender = recommender;
      this.userID = userID;
      this.recommendationTime = recommendationTime;
    }

    @Override
    public Object call() throws Exception {
      long start = System.currentTimeMillis();
      recommender.recommend(userID, 10);
      long end = System.currentTimeMillis();
      recommendationTime.addDatum(end - start);
      return null;
    }
  }

}