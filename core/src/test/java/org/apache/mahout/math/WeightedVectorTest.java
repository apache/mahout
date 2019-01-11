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
import org.junit.Test;


public class WeightedVectorTest extends AbstractVectorTest {
  @Test
  public void testLength() {
    Vector v = new DenseVector(new double[]{0.9921337470551008, 1.0031004325833064, 0.9963963182745947});
    Centroid c = new Centroid(3, new DenseVector(v), 2);
    assertEquals(c.getVector().getLengthSquared(), c.getLengthSquared(), 1.0e-17);
    // previously, this wouldn't clear the cached squared length value correctly which would cause bad distances
    c.set(0, -1);
    System.out.printf("c = %.9f\nv = %.9f\n", c.getLengthSquared(), c.getVector().getLengthSquared());
    assertEquals(c.getVector().getLengthSquared(), c.getLengthSquared(), 1.0e-17);
  }

  @Override
  public Vector vectorToTest(int size) {
    return new WeightedVector(new DenseVector(size), 4.52, 345);
  }

  @Test
  public void testOrdering() {
    WeightedVector v1 = new WeightedVector(new DenseVector(new double[]{1, 2, 3}), 5.41, 31);
    WeightedVector v2 = new WeightedVector(new DenseVector(new double[]{1, 2, 3}), 5.00, 31);
    WeightedVector v3 = new WeightedVector(new DenseVector(new double[]{1, 3, 3}), 5.00, 31);
    WeightedVector v4 = v1.clone();
    WeightedVectorComparator comparator = new WeightedVectorComparator();

    assertTrue(comparator.compare(v1, v2) > 0);
    assertTrue(comparator.compare(v3, v1) < 0);
    assertTrue(comparator.compare(v3, v2) > 0);
    assertEquals(0, comparator.compare(v4, v1));
    assertEquals(0, comparator.compare(v1, v1));
  }

  @Test
  public void testProjection() {
    Vector v1 = new DenseVector(10).assign(Functions.random());
    WeightedVector v2 = new WeightedVector(v1, v1, 31);
    assertEquals(v1.dot(v1), v2.getWeight(), 1.0e-13);
    assertEquals(31, v2.getIndex());

    Matrix y = new DenseMatrix(10, 4).assign(Functions.random());
    Matrix q = new QRDecomposition(y.viewPart(0, 10, 0, 3)).getQ();

    Vector nullSpace = y.viewColumn(3).minus(q.times(q.transpose().times(y.viewColumn(3))));

    WeightedVector v3 = new WeightedVector(q.viewColumn(0).plus(q.viewColumn(1)), nullSpace, 1);
    assertEquals(0, v3.getWeight(), 1.0e-13);

    Vector qx = q.viewColumn(0).plus(q.viewColumn(1)).normalize();
    WeightedVector v4 = new WeightedVector(qx, q.viewColumn(0), 2);
    assertEquals(Math.sqrt(0.5), v4.getWeight(), 1.0e-13);

    WeightedVector v5 = WeightedVector.project(q.viewColumn(0), qx);
    assertEquals(Math.sqrt(0.5), v5.getWeight(), 1.0e-13);
  }

  @Override
  public void testSize() {
    assertEquals("size", 3, getTestVector().getNumNonZeroElements());
  }

  @Override
  Vector generateTestVector(int cardinality) {
    return new WeightedVector(new DenseVector(cardinality), 3.14, 53);
  }
}
