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

package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.junit.Test;

public final class InvertedRunningAverageTest extends TasteTestCase {

  @Test
  public void testAverage() {
    RunningAverage avg = new FullRunningAverage();
    RunningAverage inverted = new InvertedRunningAverage(avg);
    assertEquals(0, inverted.getCount());
    avg.addDatum(1.0);
    assertEquals(1, inverted.getCount());
    assertEquals(-1.0, inverted.getAverage(), EPSILON);
    avg.addDatum(2.0);
    assertEquals(2, inverted.getCount());
    assertEquals(-1.5, inverted.getAverage(), EPSILON);
  }

  @Test
  public void testUnsupported() {
    RunningAverage avg = new FullRunningAverage();
    RunningAverage inverted = new InvertedRunningAverage(avg);
    try {
      inverted.addDatum(1.0);
      fail("Should have thrown exception");
    } catch (UnsupportedOperationException uoe) {
      // good
    }
    try {
      inverted.changeDatum(1.0);
      fail("Should have thrown exception");
    } catch (UnsupportedOperationException uoe) {
      // good
    }
    try {
      inverted.removeDatum(1.0);
      fail("Should have thrown exception");
    } catch (UnsupportedOperationException uoe) {
      // good
    }
  }

  @Test
  public void testAverageAndStdDev() {
    RunningAverageAndStdDev avg = new FullRunningAverageAndStdDev();
    RunningAverageAndStdDev inverted = new InvertedRunningAverageAndStdDev(avg);
    assertEquals(0, inverted.getCount());
    avg.addDatum(1.0);
    assertEquals(1, inverted.getCount());
    assertEquals(-1.0, inverted.getAverage(), EPSILON);
    avg.addDatum(2.0);
    assertEquals(2, inverted.getCount());
    assertEquals(-1.5, inverted.getAverage(), EPSILON);
    assertEquals(Math.sqrt(2.0)/2.0, inverted.getStandardDeviation(), EPSILON);
  }

  @Test
  public void testAndStdDevUnsupported() {
    RunningAverageAndStdDev avg = new FullRunningAverageAndStdDev();
    RunningAverage inverted = new InvertedRunningAverageAndStdDev(avg);
    try {
      inverted.addDatum(1.0);
      fail("Should have thrown exception");
    } catch (UnsupportedOperationException uoe) {
      // good
    }
    try {
      inverted.changeDatum(1.0);
      fail("Should have thrown exception");
    } catch (UnsupportedOperationException uoe) {
      // good
    }
    try {
      inverted.removeDatum(1.0);
      fail("Should have thrown exception");
    } catch (UnsupportedOperationException uoe) {
      // good
    }
  }

}