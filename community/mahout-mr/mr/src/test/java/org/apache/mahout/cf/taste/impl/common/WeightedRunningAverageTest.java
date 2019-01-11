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

/**
 * <p>Tests {@link WeightedRunningAverage} and {@link WeightedRunningAverageAndStdDev}.</p>
 */
public final class WeightedRunningAverageTest extends TasteTestCase {

  @Test
  public void testWeighted() {

    WeightedRunningAverage runningAverage = new WeightedRunningAverage();

    assertEquals(0, runningAverage.getCount());
    assertTrue(Double.isNaN(runningAverage.getAverage()));
    runningAverage.addDatum(1.0, 2.0);
    assertEquals(1.0, runningAverage.getAverage(), EPSILON);
    runningAverage.addDatum(1.0);
    assertEquals(1.0, runningAverage.getAverage(), EPSILON);
    runningAverage.addDatum(8.0, 0.5);
    assertEquals(2.0, runningAverage.getAverage(), EPSILON);
    runningAverage.addDatum(-4.0);
    assertEquals(2.0/3.0, runningAverage.getAverage(), EPSILON);

    runningAverage.removeDatum(-4.0);
    assertEquals(2.0, runningAverage.getAverage(), EPSILON);
    runningAverage.removeDatum(2.0, 2.0);
    assertEquals(2.0, runningAverage.getAverage(), EPSILON);

    runningAverage.changeDatum(0.0);
    assertEquals(2.0, runningAverage.getAverage(), EPSILON);
    runningAverage.changeDatum(4.0, 0.5);
    assertEquals(5.0/1.5, runningAverage.getAverage(), EPSILON);
  }

  @Test
  public void testWeightedAndStdDev() {

    WeightedRunningAverageAndStdDev runningAverage = new WeightedRunningAverageAndStdDev();

    assertEquals(0, runningAverage.getCount());
    assertTrue(Double.isNaN(runningAverage.getAverage()));
    assertTrue(Double.isNaN(runningAverage.getStandardDeviation()));

    runningAverage.addDatum(1.0);
    assertEquals(1.0, runningAverage.getAverage(), EPSILON);
    assertTrue(Double.isNaN(runningAverage.getStandardDeviation()));
    runningAverage.addDatum(1.0, 2.0);
    assertEquals(1.0, runningAverage.getAverage(), EPSILON);
    assertEquals(0.0, runningAverage.getStandardDeviation(), EPSILON);
    runningAverage.addDatum(8.0, 0.5);
    assertEquals(2.0, runningAverage.getAverage(), EPSILON);
    assertEquals(Math.sqrt(10.5), runningAverage.getStandardDeviation(), EPSILON);
    runningAverage.addDatum(-4.0);
    assertEquals(2.0/3.0, runningAverage.getAverage(), EPSILON);
    assertEquals(Math.sqrt(15.75), runningAverage.getStandardDeviation(), EPSILON);

    runningAverage.removeDatum(-4.0);
    assertEquals(2.0, runningAverage.getAverage(), EPSILON);
    assertEquals(Math.sqrt(10.5), runningAverage.getStandardDeviation(), EPSILON);
    runningAverage.removeDatum(2.0, 2.0);
    assertEquals(2.0, runningAverage.getAverage(), EPSILON);
    assertEquals(Math.sqrt(31.5), runningAverage.getStandardDeviation(), EPSILON);
  }

}
