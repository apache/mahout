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

package org.apache.mahout.math;

import org.junit.Test;

public final class TestOrderedIntDoubleMapping extends MahoutTestCase {

  @Test
  public void testGetSet() {

    OrderedIntDoubleMapping mapping = new OrderedIntDoubleMapping(1);

    assertEquals(0, mapping.getNumMappings());
    assertEquals(0.0, mapping.get(0), EPSILON);
    assertEquals(0.0, mapping.get(1), EPSILON);

    mapping.set(0, 1.1);
    assertEquals(1, mapping.getNumMappings());
    assertEquals(1.1, mapping.get(0), EPSILON);
    assertEquals(0.0, mapping.get(1), EPSILON);

    mapping.set(5, 6.6);
    assertEquals(2, mapping.getNumMappings());
    assertEquals(1.1, mapping.get(0), EPSILON);
    assertEquals(0.0, mapping.get(1), EPSILON);
    assertEquals(6.6, mapping.get(5), EPSILON);
    assertEquals(0.0, mapping.get(6), EPSILON);

    mapping.set(0, 0.0);
    assertEquals(1, mapping.getNumMappings());
    assertEquals(0.0, mapping.get(0), EPSILON);
    assertEquals(0.0, mapping.get(1), EPSILON);
    assertEquals(6.6, mapping.get(5), EPSILON);

    mapping.set(5, 0.0);
    assertEquals(0, mapping.getNumMappings());
    assertEquals(0.0, mapping.get(0), EPSILON);
    assertEquals(0.0, mapping.get(1), EPSILON);
    assertEquals(0.0, mapping.get(5), EPSILON);
  }

  @Test
  public void testClone() throws Exception {
    OrderedIntDoubleMapping mapping = new OrderedIntDoubleMapping(1);
    mapping.set(0, 1.1);
    mapping.set(5, 6.6);
    OrderedIntDoubleMapping clone = mapping.clone();
    assertEquals(2, clone.getNumMappings());
    assertEquals(1.1, clone.get(0), EPSILON);
    assertEquals(0.0, clone.get(1), EPSILON);
    assertEquals(6.6, clone.get(5), EPSILON);
    assertEquals(0.0, clone.get(6), EPSILON);
  }

}
