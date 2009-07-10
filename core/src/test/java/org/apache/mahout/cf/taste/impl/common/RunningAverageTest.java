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

/** <p>Tests {@link FullRunningAverage}.</p> */
public final class RunningAverageTest extends TasteTestCase {

  public void testFull() {
    doTestRunningAverage(new FullRunningAverage());
  }

  public void testCompact() {
    doTestRunningAverage(new CompactRunningAverage());
  }

  private static void doTestRunningAverage(RunningAverage runningAverage) {

    assertEquals(0, runningAverage.getCount());
    assertTrue(Double.isNaN(runningAverage.getAverage()));
    runningAverage.addDatum(1.0);
    assertEquals(1, runningAverage.getCount());
    assertEquals(1.0, runningAverage.getAverage());
    runningAverage.addDatum(1.0);
    assertEquals(2, runningAverage.getCount());
    assertEquals(1.0, runningAverage.getAverage());
    runningAverage.addDatum(4.0);
    assertEquals(3, runningAverage.getCount());
    assertEquals(2.0, runningAverage.getAverage());
    runningAverage.addDatum(-4.0);
    assertEquals(4, runningAverage.getCount());
    assertEquals(0.5, runningAverage.getAverage());

    runningAverage.removeDatum(-4.0);
    assertEquals(3, runningAverage.getCount());
    assertEquals(2.0, runningAverage.getAverage());
    runningAverage.removeDatum(4.0);
    assertEquals(2, runningAverage.getCount());
    assertEquals(1.0, runningAverage.getAverage());

    runningAverage.changeDatum(0.0);
    assertEquals(2, runningAverage.getCount());
    assertEquals(1.0, runningAverage.getAverage());
    runningAverage.changeDatum(2.0);
    assertEquals(2, runningAverage.getCount());
    assertEquals(2.0, runningAverage.getAverage());
  }

}
