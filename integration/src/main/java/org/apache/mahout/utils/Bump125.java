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

package org.apache.mahout.utils;

/**
 * Helps with making nice intervals at arbitrary scale.
 *
 * One use case is where we are producing progress or error messages every time an incoming
 * record is received.  It is generally bad form to produce a message for <i>every</i> input
 * so it would be better to produce a message for each of the first 10 records, then every
 * other record up to 20 and then every 5 records up to 50 and then every 10 records up to 100,
 * more or less. The pattern can now repeat scaled up by 100.  The total number of messages will scale
 * with the log of the number of input lines which is much more survivable than direct output
 * and because early records all get messages, we get indications early.
 */
public class Bump125 {
  private static final int[] BUMPS = {1, 2, 5};

  static int scale(double value, double base) {
    double scale = value / base;
    // scan for correct step
    int i = 0;
    while (i < BUMPS.length - 1 && BUMPS[i + 1] <= scale) {
      i++;
    }
    return BUMPS[i];
  }

  static long base(double value) {
    return Math.max(1, (long) Math.pow(10, (int) Math.floor(Math.log10(value))));
  }

  private long counter = 0;

  public long increment() {
    long delta;
    if (counter >= 10) {
      long base = base(counter / 4.0);
      int scale = scale(counter / 4.0, base);
      delta = base * scale;
    } else {
      delta = 1;
    }
    counter += delta;
    return counter;
  }
}
