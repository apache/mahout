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

package org.apache.mahout.utils.vectors;

import com.google.common.collect.Iterables;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.MahoutTestCase;
import org.junit.Test;

public final class VectorHelperTest extends MahoutTestCase {

  @Test
  public void testJsonFormatting() throws Exception {
    Vector v = new SequentialAccessSparseVector(10);
    v.set(2, 3.1);
    v.set(4, 1.0);
    v.set(6, 8.1);
    v.set(7, -100);
    v.set(9, 12.2);
    String UNUSED = "UNUSED";
    String[] dictionary = {
        UNUSED, UNUSED, "two", UNUSED, "four", UNUSED, "six", "seven", UNUSED, "nine"
    };

    assertEquals("sorted json form incorrect: ", "{nine:12.2,six:8.1,two:3.1}",
        VectorHelper.vectorToJson(v, dictionary, 3, true));
    assertEquals("unsorted form incorrect: ", "{two:3.1,four:1.0}",
        VectorHelper.vectorToJson(v, dictionary, 2, false));
    assertEquals("sorted json form incorrect: ", "{nine:12.2,six:8.1,two:3.1,four:1.0}",
        VectorHelper.vectorToJson(v, dictionary, 4, true));
    assertEquals("sorted json form incorrect: ", "{nine:12.2,six:8.1,two:3.1,four:1.0,seven:-100.0}",
        VectorHelper.vectorToJson(v, dictionary, 5, true));
    assertEquals("sorted json form incorrect: ", "{nine:12.2,six:8.1}",
        VectorHelper.vectorToJson(v, dictionary, 2, true));
    assertEquals("unsorted form incorrect: ", "{two:3.1,four:1.0}",
        VectorHelper.vectorToJson(v, dictionary, 2, false));
  }

  @Test
  public void testTopEntries() throws Exception {
    Vector v = new SequentialAccessSparseVector(10);
    v.set(2, 3.1);
    v.set(4, 1.0);
    v.set(6, 8.1);
    v.set(7, -100);
    v.set(9, 12.2);
    v.set(1, 0.0);
    v.set(3, 0.0);
    v.set(8, 2.7);
    assertEquals(6, VectorHelper.topEntries(v, 6).size());
    // when sizeOfNonZeroElementsInVector < maxEntries
    assertTrue(VectorHelper.topEntries(v, 9).size() < 9);
    // when sizeOfNonZeroElementsInVector > maxEntries
    assertTrue(VectorHelper.topEntries(v, 5).size() < Iterables.size(v.nonZeroes()));
  }

}
