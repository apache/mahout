/*
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

import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

final class StatsCallable implements Callable<Void> {
  
  private static final Logger log = LoggerFactory.getLogger(StatsCallable.class);
  
  private final Callable<Void> delegate;
  private final boolean logStats;
  private final RunningAverageAndStdDev timing;
  private final AtomicInteger noEstimateCounter;
  
  StatsCallable(Callable<Void> delegate,
                boolean logStats,
                RunningAverageAndStdDev timing,
                AtomicInteger noEstimateCounter) {
    this.delegate = delegate;
    this.logStats = logStats;
    this.timing = timing;
    this.noEstimateCounter = noEstimateCounter;
  }
  
  @Override
  public Void call() throws Exception {
    long start = System.currentTimeMillis();
    delegate.call();
    long end = System.currentTimeMillis();
    timing.addDatum(end - start);
    if (logStats) {
      Runtime runtime = Runtime.getRuntime();
      int average = (int) timing.getAverage();
      log.info("Average time per recommendation: {}ms", average);
      long totalMemory = runtime.totalMemory();
      long memory = totalMemory - runtime.freeMemory();
      log.info("Approximate memory used: {}MB / {}MB", memory / 1000000L, totalMemory / 1000000L);
      log.info("Unable to recommend in {} cases", noEstimateCounter.get());
    }
    return null;
  }

}
