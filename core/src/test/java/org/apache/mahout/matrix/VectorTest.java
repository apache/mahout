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

package org.apache.mahout.matrix;

import junit.framework.TestCase;

import java.util.Date;
import java.util.Random;

public class VectorTest extends TestCase {

  public VectorTest(String s) {
    super(s);
  }

  public void testSparseVector() throws Exception {
    SparseVector vec1 = new SparseVector(3);
    SparseVector vec2 = new SparseVector(3);
    doTestVectors(vec1, vec2);
  }

  private static void doTestVectors(Vector left, Vector right) {
    left.setQuick(0, 1);
    left.setQuick(1, 2);
    left.setQuick(2, 3);
    right.setQuick(0, 4);
    right.setQuick(1, 5);
    right.setQuick(2, 6);
    double result = left.dot(right);
    assertEquals(result + " does not equal: " + 32, 32.0, result);
  }

  public void testNormalize() throws Exception {
    SparseVector vec1 = new SparseVector(3);

    vec1.setQuick(0, 1);
    vec1.setQuick(1, 2);
    vec1.setQuick(2, 3);
    Vector norm = vec1.normalize();
    assertTrue("norm1 is null and it shouldn't be", norm != null);
    Vector expected = new SparseVector(3);

    expected.setQuick(0, 0.2672612419124244);
    expected.setQuick(1, 0.5345224838248488);
    expected.setQuick(2, 0.8017837257372732);
    assertTrue("norm is not equal to expected", norm.equals(expected));

    norm = vec1.normalize(2);
    assertTrue("norm is not equal to expected", norm.equals(expected));

    norm = vec1.normalize(1);
    expected.setQuick(0, 1.0/6);
    expected.setQuick(1, 2.0/6);
    expected.setQuick(2, 3.0/6);
    assertTrue("norm is not equal to expected", norm.equals(expected));
    norm = vec1.normalize(3);
    //TODO this is not used
    expected = vec1.times(vec1).times(vec1);

    //double sum = expected.zSum();
    //cube = Math.pow(sum, 1.0/3);
    double cube = Math.pow(36, 1.0/3);
    expected = vec1.divide(cube);
    
    assertTrue("norm: " + norm.asFormatString() + " is not equal to expected: " + expected.asFormatString(), norm.equals(expected));

    norm = vec1.normalize(Double.POSITIVE_INFINITY);
    //The max is 3, so we divide by that.
    expected.setQuick(0, 1.0/3);
    expected.setQuick(1, 2.0/3);
    expected.setQuick(2, 3.0/3);
    assertTrue("norm: " + norm.asFormatString() + " is not equal to expected: " + expected.asFormatString(), norm.equals(expected));

    norm = vec1.normalize(0);
    //The max is 3, so we divide by that.
    expected.setQuick(0, 1.0/3);
    expected.setQuick(1, 2.0/3);
    expected.setQuick(2, 3.0/3);
    assertTrue("norm: " + norm.asFormatString() + " is not equal to expected: " + expected.asFormatString(), norm.equals(expected));

    try {
      vec1.normalize(-1);
      assertTrue(false);
    } catch (IllegalArgumentException e) {
      //expected
    }

  }

  public void testMax() throws Exception {
    SparseVector vec1 = new SparseVector(3);

    vec1.setQuick(0, 1);
    vec1.setQuick(1, 3);
    vec1.setQuick(2, 2);

    double max = vec1.maxValue();
    assertTrue(max + " does not equal: " + 3, max == 3);

    int idx = vec1.maxValueIndex();
    assertTrue(idx + " does not equal: " + 1, idx == 1);

  }

  public void testDenseVector() throws Exception {
    DenseVector vec1 = new DenseVector(3);
    DenseVector vec2 = new DenseVector(3);
    doTestVectors(vec1, vec2);

  }

  public void testVectorView() throws Exception {
    SparseVector vec1 = new SparseVector(3);
    SparseVector vec2 = new SparseVector(6);
    VectorView vecV1 = new VectorView(vec1, 0, 3);
    VectorView vecV2 = new VectorView(vec2, 2, 3);
    doTestVectors(vecV1, vecV2);
  }

  /**
   * Asserts a vector using enumeration equals a given dense vector
   */
  private static void doTestEnumeration(double[] apriori, Vector vector) {
    double[] test = new double[apriori.length];
    for (Vector.Element e : vector) {
      test[e.index()] = e.get();
    }

    for (int i = 0; i < test.length; i++) {
      assertEquals(apriori[i], test[i]);
    }
  }

  public void testEnumeration() throws Exception {
    double[] apriori = {0, 1, 2, 3, 4};

    doTestEnumeration(apriori, new VectorView(new DenseVector(new double[]{
            -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), 2, 5));

    doTestEnumeration(apriori, new DenseVector(new double[]{0, 1, 2, 3, 4}));

    SparseVector sparse = new SparseVector(5);
    sparse.set(0, 0);
    sparse.set(1, 1);
    sparse.set(2, 2);
    sparse.set(3, 3);
    sparse.set(4, 4);
    doTestEnumeration(apriori, sparse);
  }

  public void testSparseVectorTimesX() {
    Random rnd = new Random();
    Vector v1 = randomSparseVector(rnd);
    double x = rnd.nextDouble();
    long t0 = new Date().getTime();
    SparseVector.optimizeTimes = false;
    Vector rRef = null;
    for (int i = 0; i < 10; i++)
      rRef = v1.times(x);
    long t1 = new Date().getTime();
    SparseVector.optimizeTimes = true;
    Vector rOpt = null;
    for (int i = 0; i < 10; i++)
      rOpt = v1.times(x);
    long t2 = new Date().getTime();
    long tOpt = t2 - t1;
    long tRef = t1 - t0;
    assertTrue(tOpt < tRef);
    System.out.println("testSparseVectorTimesX tRef=tOpt=" + (tRef - tOpt)
            + " ms for 10 iterations");
    for (int i = 0; i < 50000; i++)
      assertEquals("i=" + i, rRef.getQuick(i), rOpt.getQuick(i));
  }

  public void testSparseVectorTimesV() {
    Random rnd = new Random();
    Vector v1 = randomSparseVector(rnd);
    Vector v2 = randomSparseVector(rnd);
    long t0 = new Date().getTime();
    SparseVector.optimizeTimes = false;
    Vector rRef = null;
    for (int i = 0; i < 10; i++)
      rRef = v1.times(v2);
    long t1 = new Date().getTime();
    SparseVector.optimizeTimes = true;
    Vector rOpt = null;
    for (int i = 0; i < 10; i++)
      rOpt = v1.times(v2);
    long t2 = new Date().getTime();
    long tOpt = t2 - t1;
    long tRef = t1 - t0;
    assertTrue(tOpt < tRef);
    System.out.println("testSparseVectorTimesV tRef=tOpt=" + (tRef - tOpt)
            + " ms for 10 iterations");
    for (int i = 0; i < 50000; i++)
      assertEquals("i=" + i, rRef.getQuick(i), rOpt.getQuick(i));
  }

  private Vector randomSparseVector(Random rnd) {
    SparseVector v1 = new SparseVector(50000);
    for (int i = 0; i < 1000; i++)
      v1.setQuick((int) (rnd.nextDouble() * 50000), rnd.nextDouble());
    return v1;
  }

}
