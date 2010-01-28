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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import junit.framework.TestCase;

import static org.apache.mahout.math.function.Functions.*;

import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;

public class VectorTest extends TestCase {

  public VectorTest(String s) {
    super(s);
  }

  public void testSparseVector() throws Exception {
    RandomAccessSparseVector vec1 = new RandomAccessSparseVector(3);
    RandomAccessSparseVector vec2 = new RandomAccessSparseVector(3);
    doTestVectors(vec1, vec2);
  }

  public void testEquivalent() throws Exception {
    //names are not used for equivalent
    RandomAccessSparseVector randomAccessLeft = new RandomAccessSparseVector("foo", 3);
    SequentialAccessSparseVector sequentialAccessLeft = new SequentialAccessSparseVector("foo", 3);
    DenseVector right = new DenseVector("foo", 3);
    randomAccessLeft.setQuick(0, 1);
    randomAccessLeft.setQuick(1, 2);
    randomAccessLeft.setQuick(2, 3);
    sequentialAccessLeft.setQuick(0,1);
    sequentialAccessLeft.setQuick(1,2);
    sequentialAccessLeft.setQuick(2,3);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    right.setQuick(2, 3);
    assertTrue("equivalent didn't work", AbstractVector.equivalent(randomAccessLeft, right));
    assertTrue("equivalent didn't work", AbstractVector.equivalent(sequentialAccessLeft, right));
    assertTrue("equivalent didn't work", AbstractVector.equivalent(sequentialAccessLeft, randomAccessLeft));
    assertEquals("equals didn't work", randomAccessLeft, right);
    assertEquals("equals didn't work", sequentialAccessLeft, right);
    assertEquals("equals didn't work", sequentialAccessLeft, randomAccessLeft);
    assertFalse("equivalent didn't work", AbstractVector.strictEquivalence(randomAccessLeft, right));
    assertFalse("equivalent didn't work", AbstractVector.strictEquivalence(randomAccessLeft, sequentialAccessLeft));
    assertFalse("equivalent didn't work", AbstractVector.strictEquivalence(sequentialAccessLeft, right));

    DenseVector leftBar = new DenseVector("bar", 3);
    leftBar.setQuick(0, 1);
    leftBar.setQuick(1, 2);
    leftBar.setQuick(2, 3);
    assertTrue("equivalent didn't work", AbstractVector.equivalent(leftBar, right));
    assertFalse("equals didn't work", leftBar.equals(right));
    assertFalse("equivalent didn't work", AbstractVector.strictEquivalence(randomAccessLeft, right));
    assertFalse("equivalent didn't work", AbstractVector.strictEquivalence(sequentialAccessLeft, right));

    RandomAccessSparseVector rightBar = new RandomAccessSparseVector("bar", 3);
    rightBar.setQuick(0, 1);
    rightBar.setQuick(1, 2);
    rightBar.setQuick(2, 3);
    assertTrue("equivalent didn't work", AbstractVector.equivalent(randomAccessLeft, rightBar));
    assertFalse("equals didn't work", randomAccessLeft.equals(rightBar));
    assertFalse("equivalent didn't work", AbstractVector.strictEquivalence(randomAccessLeft, rightBar));

    right.setQuick(2, 4);
    assertFalse("equivalent didn't work", AbstractVector.equivalent(randomAccessLeft, right));
    assertFalse("equals didn't work", randomAccessLeft.equals(right));
    right = new DenseVector(4);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    right.setQuick(2, 3);
    right.setQuick(3, 3);
    assertFalse("equivalent didn't work", AbstractVector.equivalent(randomAccessLeft, right));
    assertFalse("equals didn't work", randomAccessLeft.equals(right));
    randomAccessLeft = new RandomAccessSparseVector(2);
    randomAccessLeft.setQuick(0, 1);
    randomAccessLeft.setQuick(1, 2);
    assertFalse("equivalent didn't work", AbstractVector.equivalent(randomAccessLeft, right));
    assertFalse("equals didn't work", randomAccessLeft.equals(right));

    DenseVector dense = new DenseVector(3);
    right = new DenseVector(3);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    right.setQuick(2, 3);
    dense.setQuick(0, 1);
    dense.setQuick(1, 2);
    dense.setQuick(2, 3);
    assertTrue("equivalent didn't work", AbstractVector.equivalent(dense, right));
    assertTrue("equals didn't work", dense.equals(right));

    RandomAccessSparseVector sparse = new RandomAccessSparseVector(3);
    randomAccessLeft = new RandomAccessSparseVector(3);
    sparse.setQuick(0, 1);
    sparse.setQuick(1, 2);
    sparse.setQuick(2, 3);
    randomAccessLeft.setQuick(0, 1);
    randomAccessLeft.setQuick(1, 2);
    randomAccessLeft.setQuick(2, 3);
    assertTrue("equivalent didn't work", AbstractVector.equivalent(sparse, randomAccessLeft));
    assertTrue("equals didn't work", randomAccessLeft.equals(sparse));

    VectorView v1 = new VectorView(randomAccessLeft, 0, 2);
    VectorView v2 = new VectorView(right, 0, 2);
    assertTrue("equivalent didn't work", AbstractVector.equivalent(v1, v2));
    assertTrue("equals didn't work", v1.equals(v2));
    sparse = new RandomAccessSparseVector(2);
    sparse.setQuick(0, 1);
    sparse.setQuick(1, 2);
    assertTrue("equivalent didn't work", AbstractVector.equivalent(v1, sparse));
    assertTrue("equals didn't work", v1.equals(sparse));

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
    String formattedString = left.asFormatString();
    System.out.println("Vec: " + formattedString);
    Vector vec = AbstractVector.decodeVector(formattedString);
    assertNotNull("vec is null and it shouldn't be", vec);
    assertTrue("Vector could not be decoded from the formatString",
        AbstractVector.equivalent(vec, left));
  }

  public void testGetDistanceSquared() throws Exception {
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

  private void setUpV(Vector v) {
    v.setQuick(1, 2);
    v.setQuick(2, -4);
    v.setQuick(3, -9);
  }

  private void setUpW(Vector w) {
    w.setQuick(0, -5);
    w.setQuick(1, -1);
    w.setQuick(2, 9);
    w.setQuick(3, 0.1);
    w.setQuick(4, 2.1);
  }

  public void doTestGetDistanceSquared(Vector v, Vector w) throws Exception {
    double expected = v.minus(w).getLengthSquared();
    assertTrue("a.getDistanceSquared(b) != a.minus(b).getLengthSquared",
        Math.abs(expected - v.getDistanceSquared(w)) < 1e-6);
  }

  public void testNormalize() throws Exception {
    RandomAccessSparseVector vec1 = new RandomAccessSparseVector(3);

    vec1.setQuick(0, 1);
    vec1.setQuick(1, 2);
    vec1.setQuick(2, 3);
    Vector norm = vec1.normalize();
    assertNotNull("norm1 is null and it shouldn't be", norm);

    SequentialAccessSparseVector vec2 = new SequentialAccessSparseVector(3);

    vec2.setQuick(0, 1);
    vec2.setQuick(1, 2);
    vec2.setQuick(2, 3);
    Vector norm2 = vec2.normalize();
    assertNotNull("norm1 is null and it shouldn't be", norm2);

    Vector expected = new RandomAccessSparseVector(3);

    expected.setQuick(0, 0.2672612419124244);
    expected.setQuick(1, 0.5345224838248488);
    expected.setQuick(2, 0.8017837257372732);

    assertEquals("norm is not equal to expected", norm, expected);
    assertTrue("norm is not equivalent to expected", AbstractVector.equivalent(norm2, expected));

    norm = vec1.normalize(2);
    assertEquals("norm is not equal to expected", norm, expected);

    norm2 = vec2.normalize(2);
    assertTrue("norm is not equivalent to expected", AbstractVector.equivalent(norm2, expected));

    norm = vec1.normalize(1);
    norm2 = vec2.normalize(1);
    expected.setQuick(0, 1.0 / 6);
    expected.setQuick(1, 2.0 / 6);
    expected.setQuick(2, 3.0 / 6);
    assertEquals("norm is not equal to expected", norm, expected);
    assertTrue("norm is not equivalent to expected", AbstractVector.equivalent(norm2, expected));
    norm = vec1.normalize(3);
    // TODO this is not used
    expected = vec1.times(vec1).times(vec1);

    // double sum = expected.zSum();
    // cube = Math.pow(sum, 1.0/3);
    double cube = Math.pow(36, 1.0 / 3);
    expected = vec1.divide(cube);

    assertEquals("norm: " + norm.asFormatString() + " is not equal to expected: "
        + expected.asFormatString(), norm, expected);

    norm = vec1.normalize(Double.POSITIVE_INFINITY);
    norm2 = vec2.normalize(Double.POSITIVE_INFINITY);
    // The max is 3, so we divide by that.
    expected.setQuick(0, 1.0 / 3);
    expected.setQuick(1, 2.0 / 3);
    expected.setQuick(2, 3.0 / 3);
    assertEquals("norm: " + norm.asFormatString() + " is not equal to expected: "
        + expected.asFormatString(), norm, expected);
    assertEquals("norm: " + norm2.asFormatString() + " is not equal to expected: "
        + expected.asFormatString(), norm2, expected);

    norm = vec1.normalize(0);
    // The max is 3, so we divide by that.
    expected.setQuick(0, 1.0 / 3);
    expected.setQuick(1, 2.0 / 3);
    expected.setQuick(2, 3.0 / 3);
    assertEquals("norm: " + norm.asFormatString() + " is not equal to expected: "
        + expected.asFormatString(), norm, expected);

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

  public void testMax() throws Exception {
    Vector vec1 = new RandomAccessSparseVector(3);

    vec1.setQuick(0, -1);
    vec1.setQuick(1, -3);
    vec1.setQuick(2, -2);

    double max = vec1.maxValue();
    assertEquals(max + " does not equal: " + -1, -1, max, 0.0);

    int idx = vec1.maxValueIndex();
    assertEquals(idx + " does not equal: " + 0, 0, idx);

    vec1 = new RandomAccessSparseVector(3);
    max = vec1.maxValue();
    assertEquals(max + " does not equal 0", 0d, max);

    vec1 = new DenseVector(3);
    max = vec1.maxValue();
    assertEquals(max + " does not equal 0", 0d, max);

    vec1 = new SequentialAccessSparseVector(3);
    max = vec1.maxValue();
    assertEquals(max + " does not equal 0", 0d, max);

    vec1 = new RandomAccessSparseVector(0);
    max = vec1.maxValue();
    assertEquals(max + " does not equal -inf", Double.NEGATIVE_INFINITY, max);

    vec1 = new DenseVector(0);
    max = vec1.maxValue();
    assertEquals(max + " does not equal -inf", Double.NEGATIVE_INFINITY, max);

    vec1 = new SequentialAccessSparseVector(0);
    max = vec1.maxValue();
    assertEquals(max + " does not equal -inf", Double.NEGATIVE_INFINITY, max);

  }

  public void testDenseVector() throws Exception {
    DenseVector vec1 = new DenseVector(3);
    DenseVector vec2 = new DenseVector(3);
    doTestVectors(vec1, vec2);

  }

  public void testVectorView() throws Exception {
    RandomAccessSparseVector vec1 = new RandomAccessSparseVector(3);
    RandomAccessSparseVector vec2 = new RandomAccessSparseVector(6);
    SequentialAccessSparseVector vec3 = new SequentialAccessSparseVector(3);
    SequentialAccessSparseVector vec4 = new SequentialAccessSparseVector(6);
    VectorView vecV1 = new VectorView(vec1, 0, 3);
    VectorView vecV2 = new VectorView(vec2, 2, 3);
    VectorView vecV3 = new VectorView(vec3, 0, 3);
    VectorView vecV4 = new VectorView(vec4, 2, 3);
    doTestVectors(vecV1, vecV2);
    doTestVectors(vecV3, vecV4);
  }

  /** Asserts a vector using enumeration equals a given dense vector */
  private static void doTestEnumeration(double[] apriori, Vector vector) {
    double[] test = new double[apriori.length];
    Iterator<Vector.Element> iter = vector.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element e = iter.next();
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

  public void testAggregation() throws Exception {
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

  private void doTestAggregation(Vector v, Vector w) throws Exception {
    assertEquals("aggregate(plus, pow(2)) not equal to " + v.getLengthSquared(),
        v.getLengthSquared(),
        v.aggregate(plus, pow(2)));
    assertEquals("aggregate(plus, abs) not equal to " + v.norm(1),
        v.norm(1),
        v.aggregate(plus, abs));
    assertEquals("aggregate(max, abs) not equal to " + v.norm(Double.POSITIVE_INFINITY),
        v.norm(Double.POSITIVE_INFINITY),
        v.aggregate(max, abs));

    assertEquals("v.dot(w) != v.aggregate(w, plus, mult)",
        v.dot(w),
        v.aggregate(w, plus, mult));
    assertEquals("|(v-w)|^2 != v.aggregate(w, plus, chain(pow(2), minus))",
        v.minus(w).dot(v.minus(w)),
        v.aggregate(w, plus, chain(pow(2), minus)));
  }

  private void setUpFirstVector(Vector v) {
    v.setQuick(1, 2);
    v.setQuick(2, 0.5);
    v.setQuick(3, -5);
  }

  private void setUpSecondVector(Vector v) {
    v.setQuick(0, 3);
    v.setQuick(1, -1.5);
    v.setQuick(2, -5);
    v.setQuick(3, 2);
  }

  /*public void testSparseVectorTimesX() {
    RandomUtils.useTestSeed();
    Random rnd = RandomUtils.getRandom();
    Vector v1 = randomSparseVector(rnd);
    double x = rnd.nextDouble();
    long t0 = System.currentTimeMillis();
    RandomAccessSparseVector.optimizeTimes = false;
    Vector rRef = null;
    for (int i = 0; i < 10; i++)
      rRef = v1.times(x);
    long t1 = System.currentTimeMillis();
    RandomAccessSparseVector.optimizeTimes = true;
    Vector rOpt = null;
    for (int i = 0; i < 10; i++)
      rOpt = v1.times(x);
    long t2 = System.currentTimeMillis();
    long tOpt = t2 - t1;
    long tRef = t1 - t0;
    assertTrue(tOpt < tRef);
    System.out.println("testSparseVectorTimesX tRef-tOpt=" + (tRef - tOpt)
        + " ms for 10 iterations");
    for (int i = 0; i < 50000; i++)
      assertEquals("i=" + i, rRef.getQuick(i), rOpt.getQuick(i));
  }*/

  /*public void testSparseVectorTimesV() {
    RandomUtils.useTestSeed();
    Random rnd = RandomUtils.getRandom();
    Vector v1 = randomSparseVector(rnd);
    Vector v2 = randomSparseVector(rnd);
    long t0 = System.currentTimeMillis();
    RandomAccessSparseVector.optimizeTimes = false;
    Vector rRef = null;
    for (int i = 0; i < 10; i++)
      rRef = v1.times(v2);
    long t1 = System.currentTimeMillis();
    RandomAccessSparseVector.optimizeTimes = true;
    Vector rOpt = null;
    for (int i = 0; i < 10; i++)
      rOpt = v1.times(v2);
    long t2 = System.currentTimeMillis();
    long tOpt = t2 - t1;
    long tRef = t1 - t0;
    assertTrue(tOpt < tRef);
    System.out.println("testSparseVectorTimesV tRef-tOpt=" + (tRef - tOpt)
        + " ms for 10 iterations");
    for (int i = 0; i < 50000; i++)
      assertEquals("i=" + i, rRef.getQuick(i), rOpt.getQuick(i));
  }*/

  private static Vector randomSparseVector(Random rnd) {
    RandomAccessSparseVector v1 = new RandomAccessSparseVector(50000);
    for (int i = 0; i < 1000; i++) {
      v1.setQuick(rnd.nextInt(50000), rnd.nextDouble());
    }
    return v1;
  }

  public void testLabelSerializationDense() {
    double[] values = {1.1, 2.2, 3.3};
    Vector test = new DenseVector(values);
    Map<String, Integer> bindings = new HashMap<String, Integer>();
    bindings.put("Fee", 0);
    bindings.put("Fie", 1);
    bindings.put("Foe", 2);
    test.setLabelBindings(bindings);

    Type vectorType = new TypeToken<Vector>() {
    }.getType();

    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(vectorType, new JsonVectorAdapter());
    Gson gson = builder.create();
    String json = gson.toJson(test, vectorType);
    Vector test1 = gson.fromJson(json, vectorType);
    try {
      test1.get("Fee");
      fail();
    } catch (IndexException e) {
      fail();
    } catch (UnboundLabelException e) {
      assertTrue(true);
    }

  }


  public void testNameSerialization() throws Exception {
    double[] values = {1.1, 2.2, 3.3};
    Vector test = new DenseVector("foo", values);
    String formatString = test.asFormatString();

    Vector decode = AbstractVector.decodeVector(formatString);
    assertEquals("test and decode are not equal", test, decode);

    Vector noName = new DenseVector(values);
    formatString = noName.asFormatString();

    decode = AbstractVector.decodeVector(formatString);
    assertEquals("noName and decode are not equal", noName, decode);
  }

  public void testLabelSerializationSparse() {
    double[] values = {1.1, 2.2, 3.3};
    Vector test = new RandomAccessSparseVector(3);
    for (int i = 0; i < values.length; i++) {
      test.set(i, values[i]);
    }
    Map<String, Integer> bindings = new HashMap<String, Integer>();
    bindings.put("Fee", 0);
    bindings.put("Fie", 1);
    bindings.put("Foe", 2);
    test.setLabelBindings(bindings);

    Type vectorType = new TypeToken<Vector>() {
    }.getType();

    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(vectorType, new JsonVectorAdapter());
    Gson gson = builder.create();
    String json = gson.toJson(test, vectorType);
    Vector test1 = gson.fromJson(json, vectorType);
    try {
      test1.get("Fee");
      fail();
    } catch (IndexException e) {
      fail();
    } catch (UnboundLabelException e) {
      assertTrue(true);
    }
  }

  public void testLabelSet() {
    Vector test = new DenseVector(3);
    test.set("Fee", 0, 1.1);
    test.set("Fie", 1, 2.2);
    test.set("Foe", 2, 3.3);
    assertEquals("Fee", 1.1, test.get("Fee"));
    assertEquals("Fie", 2.2, test.get("Fie"));
    assertEquals("Foe", 3.3, test.get("Foe"));
  }

  public void testHashCodeEquivalence() {
    // Hash codes must be equal if the vectors are considered equal
    Vector sparseLeft = new RandomAccessSparseVector(3);
    DenseVector denseRight = new DenseVector(3);
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

    DenseVector denseLeft = new DenseVector(3);
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

    DenseVector emptyLeft = new DenseVector("foo", 0);
    Vector emptyRight = new SequentialAccessSparseVector("foo", 0);
    assertEquals(emptyLeft, emptyRight);
    assertEquals(emptyLeft.hashCode(), emptyRight.hashCode());
    emptyRight = new RandomAccessSparseVector("foo", 0);
    assertEquals(emptyLeft, emptyRight);
    assertEquals(emptyLeft.hashCode(), emptyRight.hashCode());
  }

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

}
