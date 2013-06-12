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

package org.apache.mahout.common;

import java.io.Serializable;
import java.text.DecimalFormat;

public final class TimingStatistics implements Serializable {
  private static final DecimalFormat DF = new DecimalFormat("#.##");
  private int nCalls;
  private long minTime;
  private long maxTime;
  private long sumTime;
  private long leadSumTime;
  private double sumSquaredTime;


  /** Creates a new instance of CallStats */
  public TimingStatistics() { }

  public TimingStatistics(int nCalls, long minTime, long maxTime, long sumTime, double sumSquaredTime) {
    this.nCalls = nCalls;
    this.minTime = minTime;
    this.maxTime = maxTime;
    this.sumTime = sumTime;
    this.sumSquaredTime = sumSquaredTime;
  }

  public synchronized int getNCalls() {
    return nCalls;
  }

  public synchronized long getMinTime() {
    return Math.max(0, minTime);
  }

  public synchronized long getMaxTime() {
    return maxTime;
  }

  public synchronized long getSumTime() {
    return sumTime;
  }

  public synchronized double getSumSquaredTime() {
    return sumSquaredTime;
  }

  public synchronized long getMeanTime() {
    return nCalls == 0 ? 0 : sumTime / nCalls;
  }

  public synchronized long getStdDevTime() {
    if (nCalls == 0) {
      return 0;
    }
    double mean = getMeanTime();
    double meanSquared = mean * mean;
    double meanOfSquares = sumSquaredTime / nCalls;
    double variance = meanOfSquares - meanSquared;
    if (variance < 0) {
      return 0; // might happen due to rounding error
    }
    return (long) Math.sqrt(variance);
  }

  @Override
  public synchronized String toString() {
    return '\n'
        + "nCalls = " + nCalls + ";\n"
        + "sum    = " + DF.format(sumTime / 1000000000.0) + "s;\n"
        + "min    = " + DF.format(minTime / 1000000.0) + "ms;\n"
        + "max    = " + DF.format(maxTime / 1000000.0) + "ms;\n"
        + "mean   = " + DF.format(getMeanTime() / 1000.0) + "us;\n"
        + "stdDev = " + DF.format(getStdDevTime() / 1000.0) + "us;";
  }

  /** Ignores counting the performance metrics until leadTimeIsFinished The caller should enough time for the JIT
   *  to warm up. */
  public Call newCall(long leadTimeUsec) {
    if (leadSumTime > leadTimeUsec) {
      return new Call();
    } else {
      return new LeadTimeCall();
    }
  }

  /** Ignores counting the performance metrics. The caller should enough time for the JIT to warm up. */
  public final class LeadTimeCall extends Call {

    private LeadTimeCall() { }

    @Override
    public void end() {
      long elapsed = System.nanoTime() - startTime;
      synchronized (TimingStatistics.this) {
        leadSumTime += elapsed;
      }
    }

    @Override
    public boolean end(long sumMaxUsec) {
      end();
      return false;
    }
  }

  /**
   * A call object that can update performance metrics.
   */
  public class Call {
    protected final long startTime = System.nanoTime();

    private Call() { }

    public void end() {
      long elapsed = System.nanoTime() - startTime;
      synchronized (TimingStatistics.this) {
        nCalls++;
        if (elapsed < minTime || nCalls == 1) {
          minTime = elapsed;
        }
        if (elapsed > maxTime) {
          maxTime = elapsed;
        }
        sumTime += elapsed;
        sumSquaredTime += elapsed * elapsed;
      }
    }

    /**
     * Returns true if the sumTime as reached this limit;
     */
    public boolean end(long sumMaxUsec) {
      end();
      return sumMaxUsec < sumTime;
    }
  }
}
