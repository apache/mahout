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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import junit.framework.TestCase;

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
    SparseVector vec1 = new SparseVector(3);
    SparseVector vec2 = new SparseVector(3);
    doTestVectors(vec1, vec2);
  }

  public void testEquivalent() throws Exception {
    //names are not used for equivalent
    SparseVector left = new SparseVector("foo", 3);
    DenseVector right = new DenseVector("foo", 3);
    left.setQuick(0, 1);
    left.setQuick(1, 2);
    left.setQuick(2, 3);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    right.setQuick(2, 3);
    assertTrue("equivalent didn't work", AbstractVector.equivalent(left, right));
    assertTrue("equals didn't work", left.equals(right));
    assertTrue("equivalent didn't work", AbstractVector.strictEquivalence(left, right) == false);

    right.setQuick(2, 4);
    assertTrue("equivalent didn't work",
        AbstractVector.equivalent(left, right) == false);
    assertTrue("equals didn't work", left.equals(right) == false);
    right = new DenseVector(4);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    right.setQuick(2, 3);
    right.setQuick(3, 3);
    assertTrue("equivalent didn't work",
        AbstractVector.equivalent(left, right) == false);
    assertTrue("equals didn't work", left.equals(right) == false);
    left = new SparseVector(2);
    left.setQuick(0, 1);
    left.setQuick(1, 2);
    assertTrue("equivalent didn't work",
        AbstractVector.equivalent(left, right) == false);
    assertTrue("equals didn't work", left.equals(right) == false);

    DenseVector dense = new DenseVector(3);
    right = new DenseVector(3);
    right.setQuick(0, 1);
    right.setQuick(1, 2);
    right.setQuick(2, 3);
    dense.setQuick(0, 1);
    dense.setQuick(1, 2);
    dense.setQuick(2, 3);
    assertTrue("equivalent didn't work", AbstractVector
        .equivalent(dense, right) == true);
    assertTrue("equals didn't work", dense.equals(right) == true);

    SparseVector sparse = new SparseVector(3);
    left = new SparseVector(3);
    sparse.setQuick(0, 1);
    sparse.setQuick(1, 2);
    sparse.setQuick(2, 3);
    left.setQuick(0, 1);
    left.setQuick(1, 2);
    left.setQuick(2, 3);
    assertTrue("equivalent didn't work", AbstractVector
        .equivalent(sparse, left) == true);
    assertTrue("equals didn't work", left.equals(sparse) == true);

    VectorView v1 = new VectorView(left, 0, 2);
    VectorView v2 = new VectorView(right, 0, 2);
    assertTrue("equivalent didn't work",
        AbstractVector.equivalent(v1, v2) == true);
    assertTrue("equals didn't work", v1.equals(v2) == true);
    sparse = new SparseVector(2);
    sparse.setQuick(0, 1);
    sparse.setQuick(1, 2);
    assertTrue("equivalent didn't work",
        AbstractVector.equivalent(v1, sparse) == true);
    assertTrue("equals didn't work", v1.equals(sparse) == true);

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
    assertTrue("vec is null and it shouldn't be", vec != null);
    assertTrue("Vector could not be decoded from the formatString",
        AbstractVector.equivalent(vec, left));
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
    expected.setQuick(0, 1.0 / 6);
    expected.setQuick(1, 2.0 / 6);
    expected.setQuick(2, 3.0 / 6);
    assertTrue("norm is not equal to expected", norm.equals(expected));
    norm = vec1.normalize(3);
    // TODO this is not used
    expected = vec1.times(vec1).times(vec1);

    // double sum = expected.zSum();
    // cube = Math.pow(sum, 1.0/3);
    double cube = Math.pow(36, 1.0 / 3);
    expected = vec1.divide(cube);

    assertTrue("norm: " + norm.asFormatString() + " is not equal to expected: "
        + expected.asFormatString(), norm.equals(expected));

    norm = vec1.normalize(Double.POSITIVE_INFINITY);
    // The max is 3, so we divide by that.
    expected.setQuick(0, 1.0 / 3);
    expected.setQuick(1, 2.0 / 3);
    expected.setQuick(2, 3.0 / 3);
    assertTrue("norm: " + norm.asFormatString() + " is not equal to expected: "
        + expected.asFormatString(), norm.equals(expected));

    norm = vec1.normalize(0);
    // The max is 3, so we divide by that.
    expected.setQuick(0, 1.0 / 3);
    expected.setQuick(1, 2.0 / 3);
    expected.setQuick(2, 3.0 / 3);
    assertTrue("norm: " + norm.asFormatString() + " is not equal to expected: "
        + expected.asFormatString(), norm.equals(expected));

    try {
      vec1.normalize(-1);
      assertTrue(false);
    } catch (IllegalArgumentException e) {
      // expected
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

    SparseVector sparse = new SparseVector(5);
    sparse.set(0, 0);
    sparse.set(1, 1);
    sparse.set(2, 2);
    sparse.set(3, 3);
    sparse.set(4, 4);
    doTestEnumeration(apriori, sparse);
  }

  /*public void testSparseVectorTimesX() {
    Random rnd = new Random(0xDEADBEEFL);
    Vector v1 = randomSparseVector(rnd);
    double x = rnd.nextDouble();
    long t0 = System.currentTimeMillis();
    SparseVector.optimizeTimes = false;
    Vector rRef = null;
    for (int i = 0; i < 10; i++)
      rRef = v1.times(x);
    long t1 = System.currentTimeMillis();
    SparseVector.optimizeTimes = true;
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
    Random rnd = new Random(0xDEADBEEFL);
    Vector v1 = randomSparseVector(rnd);
    Vector v2 = randomSparseVector(rnd);
    long t0 = System.currentTimeMillis();
    SparseVector.optimizeTimes = false;
    Vector rRef = null;
    for (int i = 0; i < 10; i++)
      rRef = v1.times(v2);
    long t1 = System.currentTimeMillis();
    SparseVector.optimizeTimes = true;
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
    SparseVector v1 = new SparseVector(50000);
    for (int i = 0; i < 1000; i++) {
      {
        v1.setQuick(rnd.nextInt(50000), rnd.nextDouble());
      }
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
    assertTrue("test and decode are not equal", test.equals(decode));

    Vector noName = new DenseVector(values);
    formatString = noName.asFormatString();

    decode = AbstractVector.decodeVector(formatString);
    assertTrue("noName and decode are not equal", noName.equals(decode));
  }

  public void testLabelSerializationSparse() {
    double[] values = {1.1, 2.2, 3.3};
    Vector test = new SparseVector(3);
    for (int i = 0; i < values.length; i++) {
      {
        test.set(i, values[i]);
      }
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

}
