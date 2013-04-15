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

import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.random.MultiNormal;
import org.junit.Test;

public class CentroidTest extends AbstractVectorTest<Centroid> {
  @Test
  public void testUpdate() {
    MultiNormal f = new MultiNormal(20);

    Vector a = f.sample();
    Vector b = f.sample();
    Vector c = f.sample();

    DenseVector x = new DenseVector(a);
    Centroid x1 = new Centroid(1, x);

    x1.update(new Centroid(2, new DenseVector(b)));
    Centroid x2 = new Centroid(x1);

    x1.update(c);

    // check for correct value
    Vector mean = a.plus(b).plus(c).assign(Functions.div(3));
    assertEquals(0, x1.getVector().minus(mean).norm(1), 1.0e-8);
    assertEquals(3, x1.getWeight(), 0);

    assertEquals(0, x2.minus(a.plus(b).divide(2)).norm(1), 1.0e-8);
    assertEquals(2, x2.getWeight(), 0);

    assertEquals(0, new Centroid(x1.getIndex(), x1, x1.getWeight()).minus(x1).norm(1), 1.0e-8);

    // and verify shared storage
    assertEquals(0, x.minus(x1).norm(1), 0);

    assertEquals(3, x1.getWeight(), 1.0e-8);
    assertEquals(1, x1.getIndex());
  }

  @Override
  public Centroid vectorToTest(int size) {
    return new Centroid(new WeightedVector(new DenseVector(size), 3.15, 51));
  }

  @Override
  public void testSize() {
    assertEquals("size", 3, getTestVector().getNumNonZeroElements());
  }

  @Override
  Vector generateTestVector(int cardinality) {
    return new Centroid(new WeightedVector(new DenseVector(cardinality), 3.14, 53));
  }
}
