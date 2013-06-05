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

import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Set;

import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import com.google.common.collect.Sets;

public final class VectorTest extends MahoutTestCase {

  @Test
  public void testSparseVector()  {
    Vector vec1 = new RandomAccessSparseVector(3);
    Vector vec2 = new RandomAccessSparseVector(3);
    doTestVectors(vec1, vec2);
  }

  @Test
  public void testSparseVectorFullIteration() {
    int[] index = {0, 1, 2, 3, 4, 5};
    double[] values = {1, 2, 3, 4, 5, 6};

    assertEquals(index.length, values.length);

    int n = index.length;

    Vector vector = new SequentialAccessSparseVector(n);
    for (int i = 0; i < n; i++) {
      vector.set(index[i], values[i]);
    }

    for (int i = 0; i < n; i++) {
      assertEquals(vector.get(i), values[i], EPSILON);
    }

    int elements = 0;
    for (Element ignore : vector.all()) {
      elements++;
    }
    assertEquals(n, elements);

    assertFalse(new SequentialAccessSparseVector(0).iterator().hasNext());
  }

  @Test
  public void testSparseVectorSparseIteration() {
    int[] index = {0, 1, 2, 3, 4, 5};
    double[] values = {1, 2, 3, 4, 5, 6};

    assertEquals(index.length, values.length);

    int n = index.length;

    Vector vector = new SequentialAccessSparseVector(n);
    for (int i = 0; i < n; i++) {
      vector.set(index[i], values[i]);
    }

    for (int i = 0; i < n; i++) {
      assertEquals(vector.get(i), values[i], EPSILON);
    }

    int elements = 0;
    for (Element ignored : vector.nonZeroes()) {
      elements++;
    }
    assertEquals(n, elements);

    Vector empty = new SequentialAccessSparseVector(0);
    assertFalse(empty.nonZeroes().iterator().hasNext());
  }

  @Test
  public void testEquivalent()  {
    //names are not used for equivalent
    RandomAccessSparseVector randomAccessLeft = new RandomAccessSparseVector(3);
    Vector sequentialAccessLeft = new SequentialAccessSparseVector(3);
    Vector right = new DenseVector(3);
    randomAccessLeft.setQuick(0, 1);
    randomAccessLeft.setQuick(1, 2);
    randomAccessLeft.setQuick(2, 3);
    sequentialAccessLeft.setQuick(0,1);
    sequentialAccessLeft.setQuick(1,2);
    sequentialAccessLeft.setQuick(2,3);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    right.setQuick(2, 3);
    assertEquals(randomAccessLeft, right);
    assertEquals(sequentialAccessLeft, right);
    assertEquals(sequentialAccessLeft, randomAccessLeft);

    Vector leftBar = new DenseVector(3);
    leftBar.setQuick(0, 1);
    leftBar.setQuick(1, 2);
    leftBar.setQuick(2, 3);
    assertEquals(leftBar, right);
    assertEquals(randomAccessLeft, right);
    assertEquals(sequentialAccessLeft, right);

    Vector rightBar = new RandomAccessSparseVector(3);
    rightBar.setQuick(0, 1);
    rightBar.setQuick(1, 2);
    rightBar.setQuick(2, 3);
    assertEquals(randomAccessLeft, rightBar);

    right.setQuick(2, 4);
    assertFalse(randomAccessLeft.equals(right));
    right = new DenseVector(4);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    right.setQuick(2, 3);
    right.setQuick(3, 3);
    assertFalse(randomAccessLeft.equals(right));
    randomAccessLeft = new RandomAccessSparseVector(2);
    randomAccessLeft.setQuick(0, 1);
    randomAccessLeft.setQuick(1, 2);
    assertFalse(randomAccessLeft.equals(right));

    Vector dense = new DenseVector(3);
    right = new DenseVector(3);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    right.setQuick(2, 3);
    dense.setQuick(0, 1);
    dense.setQuick(1, 2);
    dense.setQuick(2, 3);
    assertEquals(dense, right);

    RandomAccessSparseVector sparse = new RandomAccessSparseVector(3);
    randomAccessLeft = new RandomAccessSparseVector(3);
    sparse.setQuick(0, 1);
    sparse.setQuick(1, 2);
    sparse.setQuick(2, 3);
    randomAccessLeft.setQuick(0, 1);
    randomAccessLeft.setQuick(1, 2);
    randomAccessLeft.setQuick(2, 3);
    assertEquals(randomAccessLeft, sparse);

    Vector v1 = new VectorView(randomAccessLeft, 0, 2);
    Vector v2 = new VectorView(right, 0, 2);
    assertEquals(v1, v2);
    sparse = new RandomAccessSparseVector(2);
    sparse.setQuick(0, 1);
    sparse.setQuick(1, 2);
    assertEquals(v1, sparse);
  }

  private static void doTestVectors(Vector left, Vector right) {
    left.setQuick(0, 1);
    left.setQuick(1, 2);
    left.setQuick(2, 3);
    right.setQuick(0, 4);
    right.setQuick(1, 5);
    right.setQuick(2, 6);
    double result = left.dot(right);
    assertEquals(32.0, result, EPSILON);
  }

  @Test
  public void testGetDistanceSquared()  {
    Vector v = new DenseVector(5);
    Vector w = new DenseVector(5);
    setUpV(v);
    setUpW(w);
    doTestGetDistanceSquared(v, w);

    v = new RandomAccessSparseVector(5);
    w = new RandomAccessSparseVector(5);
    setUpV(v);
    setUpW(w);
    doTestGetDistanceSquared(v, w);

    v = new SequentialAccessSparseVector(5);
    w = new SequentialAccessSparseVector(5);
    setUpV(v);
    setUpW(w);
    doTestGetDistanceSquared(v, w);

  }

  @Test
  public void testAddTo() throws Exception {
    Vector v = new DenseVector(4);
    Vector w = new DenseVector(4);
    v.setQuick(0, 1);
    v.setQuick(1, 2);
    v.setQuick(2, 0);
    v.setQuick(3, 4);

    w.setQuick(0, 1);
    w.setQuick(1, 1);
    w.setQuick(2, 1);
    w.setQuick(3, 1);

    w.assign(v, Functions.PLUS);
    Vector gold = new DenseVector(new double[]{2, 3, 1, 5});
    assertEquals(w, gold);
    assertFalse(v.equals(gold));
  }


  private static void setUpV(Vector v) {
    v.setQuick(1, 2);
    v.setQuick(2, -4);
    v.setQuick(3, -9);
  }

  private static void setUpW(Vector w) {
    w.setQuick(0, -5);
    w.setQuick(1, -1);
    w.setQuick(2, 9);
    w.setQuick(3, 0.1);
    w.setQuick(4, 2.1);
  }

  private static void doTestGetDistanceSquared(Vector v, Vector w) {
    double expected = v.minus(w).getLengthSquared();
    assertEquals(expected, v.getDistanceSquared(w), 1.0e-6);
  }

  @Test
  public void testGetLengthSquared()  {
    Vector v = new DenseVector(5);
    setUpV(v);
    doTestGetLengthSquared(v);
    v = new RandomAccessSparseVector(5);
    setUpV(v);
    doTestGetLengthSquared(v);
    v = new SequentialAccessSparseVector(5);
    setUpV(v);
    doTestGetLengthSquared(v);
  }

  public static double lengthSquaredSlowly(Vector v) {
    double d = 0.0;
    for (int i = 0; i < v.size(); i++) {
      double value = v.get(i);
      d += value * value;
    }
    return d;
  }

  private static void doTestGetLengthSquared(Vector v) {
    double expected = lengthSquaredSlowly(v);
    assertEquals("v.getLengthSquared() != sum_of_squared_elements(v)", expected, v.getLengthSquared(), 0.0);

    v.set(v.size()/2, v.get(v.size()/2) + 1.0);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via set() fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.setQuick(v.size()/5, v.get(v.size()/5) + 1.0);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via setQuick() fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    for (Element e : v.nonZeroes()) {
      if (e.index() == v.size() - 2) {
        e.set(e.get() - 5.0);
      }
    }
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via dense iterator.set fails to change lengthSquared",
                 expected, v.getLengthSquared(), EPSILON);

    int i = 0;
    for (Element e : v.nonZeroes()) {
      i++;
      if (i == v.getNumNondefaultElements() - 1) {
        e.set(e.get() - 5.0);
      }
    }
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via sparse iterator.set fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.assign(3.0);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via assign(double) fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.assign(Functions.SQUARE);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via assign(square) fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.assign(new double[v.size()]);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via assign(double[]) fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.getElement(v.size()/2).set(2.5);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via v.getElement().set() fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.normalize();
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via normalize() fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.set(0, 1.5);
    v.normalize(1.0);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via normalize(double) fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.times(2.0);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via times(double) fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.times(v);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via times(vector) fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.assign(Functions.POW, 3.0);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via assign(pow, 3.0) fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);

    v.assign(v, Functions.PLUS);
    expected = lengthSquaredSlowly(v);
    assertEquals("mutation via assign(v,plus) fails to change lengthSquared", expected, v.getLengthSquared(), EPSILON);
  }

  @Test
  public void testIterator() {

    Collection<Integer> expectedIndices = Sets.newHashSet();
    int i = 1;
    while (i <= 20) {
      expectedIndices.add(i * (i + 1) / 2);
      i++;
    }

    Vector denseVector = new DenseVector(i * i);
    for (int index : expectedIndices) {
      denseVector.set(index, (double) 2 * index);
    }
    doTestIterators(denseVector, expectedIndices);

    Vector randomAccessVector = new RandomAccessSparseVector(i * i);
    for (int index : expectedIndices) {
      randomAccessVector.set(index, (double) 2 * index);
    }
    doTestIterators(randomAccessVector, expectedIndices);

    Vector sequentialVector = new SequentialAccessSparseVector(i * i);
    for (int index : expectedIndices) {
      sequentialVector.set(index, (double) 2 * index);
    }
    doTestIterators(sequentialVector, expectedIndices);
  }

  private static void doTestIterators(Vector vector, Collection<Integer> expectedIndices) {
    expectedIndices = Sets.newHashSet(expectedIndices);
    Iterator<Element> allIterator = vector.all().iterator();
    int index = 0;
    while (allIterator.hasNext()) {
      Element element = allIterator.next();
      assertEquals(index, element.index());
      if (expectedIndices.contains(index)) {
        assertEquals((double) index * 2, element.get(), EPSILON);
      } else {
        assertEquals(0.0, element.get(), EPSILON);
      }
      index++;
    }

    for (Element element : vector.nonZeroes()) {
      index = element.index();
      assertTrue(expectedIndices.contains(index));
      assertEquals((double) index * 2, element.get(), EPSILON);
      expectedIndices.remove(index);
    }
    assertTrue(expectedIndices.isEmpty());
  }

  @Test
  public void testNormalize()  {
    Vector vec1 = new RandomAccessSparseVector(3);

    vec1.setQuick(0, 1);
    vec1.setQuick(1, 2);
    vec1.setQuick(2, 3);
    Vector norm = vec1.normalize();
    assertNotNull("norm1 is null and it shouldn't be", norm);

    Vector vec2 = new SequentialAccessSparseVector(3);

    vec2.setQuick(0, 1);
    vec2.setQuick(1, 2);
    vec2.setQuick(2, 3);
    Vector norm2 = vec2.normalize();
    assertNotNull("norm1 is null and it shouldn't be", norm2);

    Vector expected = new RandomAccessSparseVector(3);

    expected.setQuick(0, 0.2672612419124244);
    expected.setQuick(1, 0.5345224838248488);
    expected.setQuick(2, 0.8017837257372732);

    assertEquals(expected, norm);

    norm = vec1.normalize(2);
    assertEquals(expected, norm);

    norm2 = vec2.normalize(2);
    assertEquals(expected, norm2);

    norm = vec1.normalize(1);
    norm2 = vec2.normalize(1);
    expected.setQuick(0, 1.0 / 6);
    expected.setQuick(1, 2.0 / 6);
    expected.setQuick(2, 3.0 / 6);
    assertEquals(expected, norm);
    assertEquals(expected, norm2);
    norm = vec1.normalize(3);
    //expected = vec1.times(vec1).times(vec1);

    // double sum = expected.zSum();
    // cube = Math.pow(sum, 1.0/3);
    double cube = Math.pow(36, 1.0 / 3);
    expected = vec1.divide(cube);

    assertEquals(norm, expected);

    norm = vec1.normalize(Double.POSITIVE_INFINITY);
    norm2 = vec2.normalize(Double.POSITIVE_INFINITY);
    // The max is 3, so we divide by that.
    expected.setQuick(0, 1.0 / 3);
    expected.setQuick(1, 2.0 / 3);
    expected.setQuick(2, 3.0 / 3);
    assertEquals(norm, expected);
    assertEquals(norm2, expected);

    norm = vec1.normalize(0);
    // The max is 3, so we divide by that.
    expected.setQuick(0, 1.0 / 3);
    expected.setQuick(1, 2.0 / 3);
    expected.setQuick(2, 3.0 / 3);
    assertEquals(norm, expected);

    try {
      vec1.normalize(-1);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    try {
      vec2.normalize(-1);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void testLogNormalize() {
    Vector vec1 = new RandomAccessSparseVector(3);

    vec1.setQuick(0, 1);
    vec1.setQuick(1, 2);
    vec1.setQuick(2, 3);
    Vector norm = vec1.logNormalize();
    assertNotNull("norm1 is null and it shouldn't be", norm);

    Vector vec2 = new SequentialAccessSparseVector(3);

    vec2.setQuick(0, 1);
    vec2.setQuick(1, 2);
    vec2.setQuick(2, 3);
    Vector norm2 = vec2.logNormalize();
    assertNotNull("norm1 is null and it shouldn't be", norm2);

    Vector expected = new DenseVector(new double[]{
      0.2672612419124244, 0.4235990463273581, 0.5345224838248488
    });

    assertVectorEquals(expected, norm, 1.0e-15);
    assertVectorEquals(expected, norm2, 1.0e-15);

    norm = vec1.logNormalize(2);
    assertVectorEquals(expected, norm, 1.0e-15);

    norm2 = vec2.logNormalize(2);
    assertVectorEquals(expected, norm2, 1.0e-15);

    try {
      vec1.logNormalize(1);
      fail("Should fail with power == 1");
    } catch (IllegalArgumentException e) {
      // expected
    }

    try {
      vec1.logNormalize(-1);
      fail("Should fail with negative power");
    } catch (IllegalArgumentException e) {
      // expected
    }

    try {
      vec2.logNormalize(Double.POSITIVE_INFINITY);
      fail("Should fail with positive infinity norm");
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  private static void assertVectorEquals(Vector expected, Vector actual, double epsilon) {
    assertEquals(expected.size(), actual.size());
    for (Element x : expected.all()) {
      assertEquals(x.get(), actual.get(x.index()), epsilon);
    }
  }

  @Test
  public void testMax()  {
    Vector vec1 = new RandomAccessSparseVector(3);

    vec1.setQuick(0, -1);
    vec1.setQuick(1, -3);
    vec1.setQuick(2, -2);

    double max = vec1.maxValue();
    assertEquals(-1.0, max, 0.0);

    int idx = vec1.maxValueIndex();
    assertEquals(0, idx);

    vec1 = new RandomAccessSparseVector(3);

    vec1.setQuick(0, -1);
    vec1.setQuick(2, -2);

    max = vec1.maxValue();
    assertEquals(0.0, max, 0.0);

    idx = vec1.maxValueIndex();
    assertEquals(1, idx);

    vec1 = new SequentialAccessSparseVector(3);

    vec1.setQuick(0, -1);
    vec1.setQuick(2, -2);

    max = vec1.maxValue();
    assertEquals(0.0, max, 0.0);

    idx = vec1.maxValueIndex();
    assertEquals(1, idx);

    vec1 = new DenseVector(3);

    vec1.setQuick(0, -1);
    vec1.setQuick(2, -2);

    max = vec1.maxValue();
    assertEquals(0.0, max, 0.0);

    idx = vec1.maxValueIndex();
    assertEquals(1, idx);

    vec1 = new RandomAccessSparseVector(3);
    max = vec1.maxValue();
    assertEquals(0.0, max, EPSILON);

    vec1 = new DenseVector(3);
    max = vec1.maxValue();
    assertEquals(0.0, max, EPSILON);

    vec1 = new SequentialAccessSparseVector(3);
    max = vec1.maxValue();
    assertEquals(0.0, max, EPSILON);

    vec1 = new RandomAccessSparseVector(0);
    max = vec1.maxValue();
    assertEquals(Double.NEGATIVE_INFINITY, max, EPSILON);

    vec1 = new DenseVector(0);
    max = vec1.maxValue();
    assertEquals(Double.NEGATIVE_INFINITY, max, EPSILON);

    vec1 = new SequentialAccessSparseVector(0);
    max = vec1.maxValue();
    assertEquals(Double.NEGATIVE_INFINITY, max, EPSILON);

  }

  @Test
  public void testMin()  {
    Vector vec1 = new RandomAccessSparseVector(3);

    vec1.setQuick(0, 1);
    vec1.setQuick(1, 3);
    vec1.setQuick(2, 2);

    double max = vec1.minValue();
    assertEquals(1.0, max, 0.0);

    int idx = vec1.maxValueIndex();
    assertEquals(1, idx);

    vec1 = new RandomAccessSparseVector(3);

    vec1.setQuick(0, -1);
    vec1.setQuick(2, -2);

    max = vec1.maxValue();
    assertEquals(0.0, max, 0.0);

    idx = vec1.maxValueIndex();
    assertEquals(1, idx);

    vec1 = new SequentialAccessSparseVector(3);

    vec1.setQuick(0, -1);
    vec1.setQuick(2, -2);

    max = vec1.maxValue();
    assertEquals(0.0, max, 0.0);

    idx = vec1.maxValueIndex();
    assertEquals(1, idx);

    vec1 = new DenseVector(3);

    vec1.setQuick(0, -1);
    vec1.setQuick(2, -2);

    max = vec1.maxValue();
    assertEquals(0.0, max, 0.0);

    idx = vec1.maxValueIndex();
    assertEquals(1, idx);

    vec1 = new RandomAccessSparseVector(3);
    max = vec1.maxValue();
    assertEquals(0.0, max, EPSILON);

    vec1 = new DenseVector(3);
    max = vec1.maxValue();
    assertEquals(0.0, max, EPSILON);

    vec1 = new SequentialAccessSparseVector(3);
    max = vec1.maxValue();
    assertEquals(0.0, max, EPSILON);

    vec1 = new RandomAccessSparseVector(0);
    max = vec1.maxValue();
    assertEquals(Double.NEGATIVE_INFINITY, max, EPSILON);

    vec1 = new DenseVector(0);
    max = vec1.maxValue();
    assertEquals(Double.NEGATIVE_INFINITY, max, EPSILON);

    vec1 = new SequentialAccessSparseVector(0);
    max = vec1.maxValue();
    assertEquals(Double.NEGATIVE_INFINITY, max, EPSILON);

  }

  @Test
  public void testDenseVector()  {
    Vector vec1 = new DenseVector(3);
    Vector vec2 = new DenseVector(3);
    doTestVectors(vec1, vec2);
  }

  @Test
  public void testVectorView()  {
    RandomAccessSparseVector vec1 = new RandomAccessSparseVector(3);
    RandomAccessSparseVector vec2 = new RandomAccessSparseVector(6);
    SequentialAccessSparseVector vec3 = new SequentialAccessSparseVector(3);
    SequentialAccessSparseVector vec4 = new SequentialAccessSparseVector(6);
    Vector vecV1 = new VectorView(vec1, 0, 3);
    Vector vecV2 = new VectorView(vec2, 2, 3);
    Vector vecV3 = new VectorView(vec3, 0, 3);
    Vector vecV4 = new VectorView(vec4, 2, 3);
    doTestVectors(vecV1, vecV2);
    doTestVectors(vecV3, vecV4);
  }

  /** Asserts a vector using enumeration equals a given dense vector */
  private static void doTestEnumeration(double[] apriori, Vector vector) {
    double[] test = new double[apriori.length];
    for (Element e : vector.all()) {
      test[e.index()] = e.get();
    }

    for (int i = 0; i < test.length; i++) {
      assertEquals(apriori[i], test[i], EPSILON);
    }
  }

  @Test
  public void testEnumeration()  {
    double[] apriori = {0, 1, 2, 3, 4};

    doTestEnumeration(apriori, new VectorView(new DenseVector(new double[]{
        -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), 2, 5));

    doTestEnumeration(apriori, new DenseVector(new double[]{0, 1, 2, 3, 4}));

    Vector sparse = new RandomAccessSparseVector(5);
    sparse.set(0, 0);
    sparse.set(1, 1);
    sparse.set(2, 2);
    sparse.set(3, 3);
    sparse.set(4, 4);
    doTestEnumeration(apriori, sparse);

    sparse = new SequentialAccessSparseVector(5);
    sparse.set(0, 0);
    sparse.set(1, 1);
    sparse.set(2, 2);
    sparse.set(3, 3);
    sparse.set(4, 4);
    doTestEnumeration(apriori, sparse);

  }

  @Test
  public void testAggregation()  {
    Vector v = new DenseVector(5);
    Vector w = new DenseVector(5);
    setUpFirstVector(v);
    setUpSecondVector(w);
    doTestAggregation(v, w);
    v = new RandomAccessSparseVector(5);
    w = new RandomAccessSparseVector(5);
    setUpFirstVector(v);
    doTestAggregation(v, w);
    setUpSecondVector(w);
    doTestAggregation(w, v);
    v = new SequentialAccessSparseVector(5);
    w = new SequentialAccessSparseVector(5);
    setUpFirstVector(v);
    doTestAggregation(v, w);
    setUpSecondVector(w);
    doTestAggregation(w, v);
  }

  private static void doTestAggregation(Vector v, Vector w) {
    assertEquals("aggregate(plus, pow(2)) not equal to " + v.getLengthSquared(),
        v.getLengthSquared(),
        v.aggregate(Functions.PLUS, Functions.pow(2)), EPSILON);
    assertEquals("aggregate(plus, abs) not equal to " + v.norm(1),
        v.norm(1),
        v.aggregate(Functions.PLUS, Functions.ABS), EPSILON);
    assertEquals("aggregate(max, abs) not equal to " + v.norm(Double.POSITIVE_INFINITY),
        v.norm(Double.POSITIVE_INFINITY),
        v.aggregate(Functions.MAX, Functions.ABS), EPSILON);

    assertEquals("v.dot(w) != v.aggregate(w, plus, mult)",
        v.dot(w),
        v.aggregate(w, Functions.PLUS, Functions.MULT), EPSILON);
    assertEquals("|(v-w)|^2 != v.aggregate(w, plus, chain(pow(2), minus))",
        v.minus(w).dot(v.minus(w)),
        v.aggregate(w, Functions.PLUS, Functions.chain(Functions.pow(2), Functions.MINUS)), EPSILON);
  }

  @Test
  public void testEmptyAggregate1() {
    assertEquals(1.0, new DenseVector(new double[]{1}).aggregate(Functions.MIN, Functions.IDENTITY), EPSILON);
    assertEquals(1.0, new DenseVector(new double[]{2, 1}).aggregate(Functions.MIN, Functions.IDENTITY), EPSILON);
    assertEquals(0, new DenseVector(new double[0]).aggregate(Functions.MIN, Functions.IDENTITY), 0);
  }

  @Test
  public void testEmptyAggregate2() {
    assertEquals(3.0, new DenseVector(new double[]{1}).aggregate(
        new DenseVector(new double[]{2}), Functions.MIN, Functions.PLUS), EPSILON);
    assertEquals(0,
        new DenseVector(new double[0]).aggregate(new DenseVector(new double[0]), Functions.MIN, Functions.PLUS), 0);
  }

  private static void setUpFirstVector(Vector v) {
    v.setQuick(1, 2);
    v.setQuick(2, 0.5);
    v.setQuick(3, -5);
  }

  private static void setUpSecondVector(Vector v) {
    v.setQuick(0, 3);
    v.setQuick(1, -1.5);
    v.setQuick(2, -5);
    v.setQuick(3, 2);
  }

  @Test
  public void testHashCodeEquivalence() {
    // Hash codes must be equal if the vectors are considered equal
    Vector sparseLeft = new RandomAccessSparseVector(3);
    Vector denseRight = new DenseVector(3);
    sparseLeft.setQuick(0, 1);
    sparseLeft.setQuick(1, 2);
    sparseLeft.setQuick(2, 3);
    denseRight.setQuick(0, 1);
    denseRight.setQuick(1, 2);
    denseRight.setQuick(2, 3);
    assertEquals(sparseLeft, denseRight);
    assertEquals(sparseLeft.hashCode(), denseRight.hashCode());

    sparseLeft = new SequentialAccessSparseVector(3);
    sparseLeft.setQuick(0, 1);
    sparseLeft.setQuick(1, 2);
    sparseLeft.setQuick(2, 3);
    assertEquals(sparseLeft, denseRight);
    assertEquals(sparseLeft.hashCode(), denseRight.hashCode());

    Vector denseLeft = new DenseVector(3);
    denseLeft.setQuick(0, 1);
    denseLeft.setQuick(1, 2);
    denseLeft.setQuick(2, 3);
    assertEquals(denseLeft, denseRight);
    assertEquals(denseLeft.hashCode(), denseRight.hashCode());

    Vector sparseRight = new SequentialAccessSparseVector(3);
    sparseRight.setQuick(0, 1);
    sparseRight.setQuick(1, 2);
    sparseRight.setQuick(2, 3);
    assertEquals(sparseLeft, sparseRight);
    assertEquals(sparseLeft.hashCode(), sparseRight.hashCode());

    DenseVector emptyLeft = new DenseVector(0);
    Vector emptyRight = new SequentialAccessSparseVector(0);
    assertEquals(emptyLeft, emptyRight);
    assertEquals(emptyLeft.hashCode(), emptyRight.hashCode());
    emptyRight = new RandomAccessSparseVector(0);
    assertEquals(emptyLeft, emptyRight);
    assertEquals(emptyLeft.hashCode(), emptyRight.hashCode());
  }

  @Test
  public void testHashCode() {
    // Make sure that hash([1,0,2]) != hash([1,2,0])
    Vector left = new SequentialAccessSparseVector(3);
    Vector right = new SequentialAccessSparseVector(3);
    left.setQuick(0, 1);
    left.setQuick(2, 2);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    assertFalse(left.equals(right));
    assertFalse(left.hashCode() == right.hashCode());

    left = new RandomAccessSparseVector(3);
    right = new RandomAccessSparseVector(3);
    left.setQuick(0, 1);
    left.setQuick(2, 2);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    assertFalse(left.equals(right));
    assertFalse(left.hashCode() == right.hashCode());

    // Make sure that hash([1,0,2,0,0,0]) != hash([1,0,2])
    right = new SequentialAccessSparseVector(5);
    right.setQuick(0, 1);
    right.setQuick(2, 2);
    assertFalse(left.equals(right));
    assertFalse(left.hashCode() == right.hashCode());

    right = new RandomAccessSparseVector(5);
    right.setQuick(0, 1);
    right.setQuick(2, 2);
    assertFalse(left.equals(right));
    assertFalse(left.hashCode() == right.hashCode());
  }

  @Test
  public void testIteratorRasv() {
    testIterator(new RandomAccessSparseVector(99));
    testEmptyAllIterator(new RandomAccessSparseVector(0));
    testExample1NonZeroIterator(new RandomAccessSparseVector(13));
  }

  @Test
  public void testIteratorSasv() {
    testIterator(new SequentialAccessSparseVector(99));
    testEmptyAllIterator(new SequentialAccessSparseVector(0));
    testExample1NonZeroIterator(new SequentialAccessSparseVector(13));
  }

  @Test
  public void testIteratorDense() {
    testIterator(new DenseVector(99));
    testEmptyAllIterator(new DenseVector(0));
    testExample1NonZeroIterator(new DenseVector(13));
  }

  private static void testIterator(Vector vector) {
    testSkips(vector.like());
    testSkipsLast(vector.like());
    testEmptyNonZeroIterator(vector.like());
    testSingleNonZeroIterator(vector.like());
  }

  private static void testSkips(Vector vector) {
    vector.set(0, 1);
    vector.set(2, 2);
    vector.set(4, 3);
    vector.set(6, 4);

    // Test non zero iterator.
    Iterator<Element> it = vector.nonZeroes().iterator();
    Element element = null;
    int i = 0;
    while (it.hasNext()) {  // hasNext is called more often than next
      if (i % 2 == 0) {
        element = it.next();
      }
      //noinspection ConstantConditions
      assertEquals(element.index(), 2* (i/2));
      assertEquals(element.get(), vector.get(2* (i/2)), 0);
      ++i;
    }
    assertEquals(7, i);  // Last element is print only once.

    // Test all iterator.
    it = vector.all().iterator();
    element = null;
    i = 0;
    while (it.hasNext()) { // hasNext is called more often than next
      if (i % 2 == 0) {
        element = it.next();
      }
      //noinspection ConstantConditions
      assertEquals(element.index(), i/2);
      assertEquals(element.get(), vector.get(i/2), 0);
      ++i;
    }
    assertEquals(197, i);  // Last element is print only once.
  }

  private static void testSkipsLast(Vector vector) {
    vector.set(1, 6);
    vector.set(98, 6);

    // Test non zero iterator.
    Iterator<Element> it = vector.nonZeroes().iterator();

    int i = 0;
    while (it.hasNext()) {  // hasNext is called more often than next
      it.next();
      ++i;
    }
    assertEquals(2, i);  // Last element is print only once.

    // Test all iterator.
    it = vector.all().iterator();
    i = 0;
    while (it.hasNext()) { // hasNext is called more often than next
      Element element = it.next();
      assertEquals(i, element.index());
      ++i;
    }
    assertFalse(it.hasNext());
    assertEquals(99, i);  // Last element is print only once.
  }

  // Test NonZeroIterator on an list with 0 elements
  private static void testEmptyNonZeroIterator(Vector vector) {
    // Test non zero iterator.
    Iterator<Element> it = vector.nonZeroes().iterator();
    int i = 0;
    while (it.hasNext()) {
      ++i;
    }
    assertEquals(0, i);

    it = vector.nonZeroes().iterator();
    assertFalse(it.hasNext());
    try {
      it.next();
      fail();
    } catch (NoSuchElementException e) {
      // expected;
    }
  }

  // Test AllIterator on an list with 0 cardinality
  private static void testEmptyAllIterator(Vector vector) {
    // Test non zero iterator.
    Iterator<Element> it = vector.all().iterator();
    int i = 0;
    while (it.hasNext()) {
      ++i;
    }
    assertEquals(0, i);

    it = vector.nonZeroes().iterator();
    assertFalse(it.hasNext());
    try {
      it.next();
      fail();
    } catch (NoSuchElementException e) {
      // expected;
    }

    it = vector.all().iterator();
    assertFalse(it.hasNext());
    try {
      it.next();
      fail();
    } catch (NoSuchElementException e) {
      // expected;
    }
  }

  // Test NonZeroIterator on an list with 1 elements
  private static void testSingleNonZeroIterator(Vector vector) {
    vector.set(1, 6);
    // Test non zero iterator.
    Iterator<Element> it = vector.nonZeroes().iterator();
    for (int i = 0; i < 10; ++i) {
      assertTrue(it.hasNext());
    }

    it = vector.nonZeroes().iterator();
    it.next();
    for (int i = 0; i < 10; ++i) {
      assertFalse(it.hasNext());
    }
    try {
      it.next();
      fail();
    } catch (NoSuchElementException e) {
      // expected;
    }
  }

  // Test NonZeroIterator on double[] { 0, 2, 0, 0, 8, 3, 0, 6, 0, 1, 1, 2, 1 }
  private static void testExample1NonZeroIterator(Vector vector) {
    double[] val = { 0, 2, 0, 0, 8, 3, 0, 6, 0, 1, 1, 2, 1 };
    for (int i = 0; i < val.length; ++i) {
      vector.set(i, val[i]);
    }

    Set<Integer> expected = Sets.newHashSet(1, 4, 5, 7, 9, 10, 11, 12);
    Set<Double> expectedValue = Sets.newHashSet(2.0, 8.0, 3.0, 6.0, 1.0);
    // Test non zero iterator.
    Iterator<Element> it = vector.nonZeroes().iterator();
    int i = 0;
    while (it.hasNext()) {
      Element e = it.next();
      assertTrue(expected.contains(e.index()));
      assertTrue(expectedValue.contains(e.get()));
      ++i;
    }
    assertEquals(8, i);

    // Check if the non zero elements are correct.
    assertEquals(8, vector.getNumNonZeroElements());

    // Set one element to 0.
    it = vector.nonZeroes().iterator();
    i = 0;
    while (it.hasNext()) {
      Element e = it.next();
      if (e.index() == 5) {
        e.set(0.0);
      }
      ++i;
    }
    assertEquals(8, i);
    assertEquals(7, vector.getNumNonZeroElements());

    // Remove one element
    it = vector.nonZeroes().iterator();
    i = 0;
    while (it.hasNext()) {
      Element e = it.next();
      if (e.index() == 5) {
        vector.set(5, 0.0);
      }
      ++i;
    }
    assertEquals(7, i); // This just got messed up.
    // TODO: throw an exception if the underlying hashmap or array length is modified.
    assertEquals(7, vector.getNumNonZeroElements());
  }
}
