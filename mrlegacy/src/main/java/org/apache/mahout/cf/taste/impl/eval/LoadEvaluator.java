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

import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.SamplingLongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * Simple helper class for running load on a Recommender.
 */
public final class LoadEvaluator {
  
  private LoadEvaluator() { }

  public static LoadStatistics runLoad(Recommender recommender) throws TasteException {
    return runLoad(recommender, 10);
  }
  
  public static LoadStatistics runLoad(Recommender recommender, int howMany) throws TasteException {
    DataModel dataModel = recommender.getDataModel();
    int numUsers = dataModel.getNumUsers();
    double sampleRate = 1000.0 / numUsers;
    LongPrimitiveIterator userSampler =
        SamplingLongPrimitiveIterator.maybeWrapIterator(dataModel.getUserIDs(), sampleRate);
    recommender.recommend(userSampler.next(), howMany); // Warm up
    Collection<Callable<Void>> callables = Lists.newArrayList();
    while (userSampler.hasNext()) {
      callables.add(new LoadCallable(recommender, userSampler.next()));
    }
    AtomicInteger noEstimateCounter = new AtomicInteger();
    RunningAverageAndStdDev timing = new FullRunningAverageAndStdDev();
    AbstractDifferenceRecommenderEvaluator.execute(callables, noEstimateCounter, timing);
    return new LoadStatistics(timing);
  }

}
