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

public final class TimingStatistics implements Serializable {

  private int nCalls;
  private long minTime;
  private long maxTime;
  private long sumTime;
  private double sumSquaredTime;

  /** Creates a new instance of CallStats */
  public TimingStatistics() {
  }

  public TimingStatistics(int nCalls, long minTime, long maxTime, long sumTime,
                          double sumSquaredTime) {
    this.nCalls = nCalls;
    this.minTime = minTime;
    this.maxTime = maxTime;
    this.sumTime = sumTime;
    this.sumSquaredTime = sumSquaredTime;
  }

  public int getNCalls() {
    return nCalls;
  }

  public long getMinTime() {
    return Math.max(0, minTime);
  }

  public long getMaxTime() {
    return maxTime;
  }

  public long getSumTime() {
    return sumTime;
  }

  public double getSumSquaredTime() {
    return sumSquaredTime;
  }

  public long getMeanTime() {
    return nCalls == 0 ? 0 : sumTime / nCalls;
  }

  public long getStdDevTime() {
    if (nCalls == 0)
      return 0;
    double mean = getMeanTime();
    double meanSquared = mean * mean;
    double meanOfSquares = sumSquaredTime / nCalls;
    double variance = meanOfSquares - meanSquared;
    if (variance < 0)
      return 0;  // might happen due to rounding error
    return (long) Math.sqrt(variance);
  }

  public String toString() {
    return '\n' +
        "nCalls = " + nCalls + ";\n" +
        "sumTime = " + sumTime / 1000000000.0f + "s;\n" +
        "minTime = " + minTime / 1000000.0f + "ms;\n" +
        "maxTime = " + maxTime / 1000000.0f + "ms;\n" +
        "meanTime = " + getMeanTime() / 1000000.0f + "ms;\n" +
        "stdDevTime = " + getStdDevTime() / 1000000.0f + "ms;";
  }

  public Call newCall() {
    return new Call();
  }

  public class Call {
    private final long startTime = System.nanoTime();

    private Call() {
    }

    public void end() {
      long elapsed = System.nanoTime() - startTime;
      synchronized (TimingStatistics.this) {
        nCalls++;
        if (elapsed < minTime || nCalls == 1)
          minTime = elapsed;
        if (elapsed > maxTime)
          maxTime = elapsed;
        sumTime += elapsed;
        double elapsedFP = elapsed;
        sumSquaredTime += elapsedFP * elapsedFP;
      }
    }
  }
}
