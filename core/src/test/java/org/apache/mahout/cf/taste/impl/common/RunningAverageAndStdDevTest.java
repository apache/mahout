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

public final class RunningAverageAndStdDevTest extends TasteTestCase {

  public void testFull() {
    doTestAverageAndStdDev(new FullRunningAverageAndStdDev());
  }

  public void testCompact() {
    doTestAverageAndStdDev(new CompactRunningAverageAndStdDev());
  }

  private static void doTestAverageAndStdDev(RunningAverageAndStdDev average) {

    assertEquals(0, average.getCount());
    assertTrue(Double.isNaN(average.getAverage()));
    assertTrue(Double.isNaN(average.getStandardDeviation()));

    average.addDatum(6.0);
    assertEquals(1, average.getCount());
    assertEquals(6.0, average.getAverage(), EPSILON);
    assertTrue(Double.isNaN(average.getStandardDeviation()));

    average.addDatum(6.0);
    assertEquals(2, average.getCount());
    assertEquals(6.0, average.getAverage(), EPSILON);
    assertEquals(0.0, average.getStandardDeviation(), EPSILON);

    average.removeDatum(6.0);
    assertEquals(1, average.getCount());
    assertEquals(6.0, average.getAverage(), EPSILON);
    assertTrue(Double.isNaN(average.getStandardDeviation()));

    average.addDatum(-4.0);
    assertEquals(2, average.getCount());
    assertEquals(1.0, average.getAverage(), EPSILON);
    assertEquals(5.0 * 1.4142135623730951, average.getStandardDeviation(), EPSILON);

    average.removeDatum(4.0);
    assertEquals(1, average.getCount());
    assertEquals(-2.0, average.getAverage(), EPSILON);
    assertTrue(Double.isNaN(average.getStandardDeviation()));

  }

}
