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

package org.apache.mahout.benchmark;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.TimingStatistics;
import org.apache.mahout.math.Vector;

import com.google.common.base.Function;

public final class BenchmarkRunner {
  private static final int BUCKET_SIZE = 10000;
  private static final Random R = RandomUtils.getRandom();
  private final long maxTimeUsec;
  private final long leadTimeUsec;

  public BenchmarkRunner(long leadTimeMs, long maxTimeMs) {
    maxTimeUsec = TimeUnit.MILLISECONDS.toNanos(maxTimeMs);
    leadTimeUsec = TimeUnit.MILLISECONDS.toNanos(leadTimeMs);
  }

  public abstract static class BenchmarkFn implements Function<Integer, Boolean> {
    protected int randIndex() {
      return BenchmarkRunner.randIndex();
    }

    protected boolean randBool() {
      return BenchmarkRunner.randBool();
    }

    /**
     * Adds a random data dependency so that JVM does not remove dead code.
     */
    protected boolean depends(Vector v) {
      return randIndex() < v.getNumNondefaultElements();
    }
  }

  public abstract static class BenchmarkFnD implements Function<Integer, Double> {
    protected int randIndex() {
      return BenchmarkRunner.randIndex();
    }

    protected boolean randBool() {
      return BenchmarkRunner.randBool();
    }

    /**
     * Adds a random data dependency so that JVM does not remove dead code.
     */
    protected boolean depends(Vector v) {
      return randIndex() < v.getNumNondefaultElements();
    }
  }

  private static int randIndex() {
    return R.nextInt(BUCKET_SIZE);
  }

  private static boolean randBool() {
    return R.nextBoolean();
  }

  public TimingStatistics benchmark(BenchmarkFn function) {
    TimingStatistics stats = new TimingStatistics();
    boolean result = false;
    while (true) {
      int i = R.nextInt(BUCKET_SIZE);
      TimingStatistics.Call call = stats.newCall(leadTimeUsec);
      result = result ^ function.apply(i);
      if (call.end(maxTimeUsec)) {
        break;
      }
    }
    return stats;
  }

  public TimingStatistics benchmarkD(BenchmarkFnD function) {
    TimingStatistics stats = new TimingStatistics();
    double result = 0;
    while (true) {
      int i = R.nextInt(BUCKET_SIZE);
      TimingStatistics.Call call = stats.newCall(leadTimeUsec);
      result += function.apply(i);
      if (call.end(maxTimeUsec)) {
        break;
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    System.err.println("Result = " + result);
    return stats;
  }
}
