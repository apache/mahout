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

package org.apache.mahout.math;

import org.junit.Test;

import static org.junit.Assert.assertEquals;


public class WeightedVectorTest {
  @Test
  public void testLength() {
    Vector v = new DenseVector(new double[]{0.9921337470551008, 1.0031004325833064, 0.9963963182745947});
    Centroid c = new Centroid(3, new DenseVector(v), 2);
    assertEquals(c.getVector().getLengthSquared(), c.getLengthSquared(), 1e-17);
    // previously, this wouldn't clear the cached squared length value correctly which would cause bad distances
    c.set(0, -1);
    System.out.printf("c = %.9f\nv = %.9f\n", c.getLengthSquared(), c.getVector().getLengthSquared());
    assertEquals(c.getVector().getLengthSquared(), c.getLengthSquared(), 1e-17);
  }
}
