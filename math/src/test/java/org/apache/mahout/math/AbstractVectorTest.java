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

import java.util.Iterator;
import java.util.Random;

import com.google.common.collect.Iterables;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.jet.random.Normal;
import org.apache.mahout.math.random.MultiNormal;
import org.junit.Before;
import org.junit.Test;

/**
 * Makes sure that a vector under test acts the same as a DenseVector or RandomAccessSparseVector
 * (according to whether it is dense or sparse).  Most operations need to be done within a reasonable
 * tolerance.
 *
 * The idea is that a new vector implementation can extend AbstractVectorTest to get pretty high
 * confidence that it is working correctly.
 */
public abstract class AbstractVectorTest<T extends Vector> extends MahoutTestCase {

  private static final double FUZZ = 1.0e-13;
  private static final double[] values = {1.1, 2.2, 3.3};
  private static final double[] gold = {0.0, 1.1, 0.0, 2.2, 0.0, 3.3, 0.0};
  private Vector test;

  private static void checkIterator(Iterator<Vector.Element> nzIter, double[] values) {
    while (nzIter.hasNext()) {
      Vector.Element elt = nzIter.next();
      assertEquals(elt.index() + " Value: " + values[elt.index()]
          + " does not equal: " + elt.get(), values[elt.index()], elt.get(), 0.0);
    }
  }

  public abstract T vectorToTest(int size);

  @Test
  public void testSimpleOps() {

    T v0 = vectorToTest(20);
    Random gen = RandomUtils.getRandom();
    Vector v1 = v0.assign(new Normal(0, 1, gen));

    // verify that v0 and v1 share and are identical
    assertEquals(v0.get(12), v1.get(12), 0);
    v0.set(12, gen.nextDouble());
    assertEquals(v0.get(12), v1.get(12), 0);
    assertSame(v0, v1);

    Vector v2 = vectorToTest(20).assign(new Normal(0, 1, gen));
    Vector dv1 = new DenseVector(v1);
    Vector dv2 = new DenseVector(v2);
    Vector sv1 = new RandomAccessSparseVector(v1);
    Vector sv2 = new RandomAccessSparseVector(v2);

    assertEquals(0, dv1.plus(dv2).getDistanceSquared(v1.plus(v2)), FUZZ);
    assertEquals(0, dv1.plus(dv2).getDistanceSquared(v1.plus(dv2)), FUZZ);
    assertEquals(0, dv1.plus(dv2).getDistanceSquared(v1.plus(sv2)), FUZZ);
    assertEquals(0, dv1.plus(dv2).getDistanceSquared(sv1.plus(v2)), FUZZ);

    assertEquals(0, dv1.times(dv2).getDistanceSquared(v1.times(v2)), FUZZ);
    assertEquals(0, dv1.times(dv2).getDistanceSquared(v1.times(dv2)), FUZZ);
    assertEquals(0, dv1.times(dv2).getDistanceSquared(v1.times(sv2)), FUZZ);
    assertEquals(0, dv1.times(dv2).getDistanceSquared(sv1.times(v2)), FUZZ);

    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.minus(v2)), FUZZ);
    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.minus(dv2)), FUZZ);
    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.minus(sv2)), FUZZ);
    assertEquals(0, dv1.minus(dv2).getDistanceSquared(sv1.minus(v2)), FUZZ);

    double z = gen.nextDouble();
    assertEquals(0, dv1.divide(z).getDistanceSquared(v1.divide(z)), 1.0e-12);
    assertEquals(0, dv1.times(z).getDistanceSquared(v1.times(z)), 1.0e-12);
    assertEquals(0, dv1.plus(z).getDistanceSquared(v1.plus(z)), 1.0e-12);

    assertEquals(dv1.dot(dv2), v1.dot(v2), FUZZ);
    assertEquals(dv1.dot(dv2), v1.dot(dv2), FUZZ);
    assertEquals(dv1.dot(dv2), v1.dot(sv2), FUZZ);
    assertEquals(dv1.dot(dv2), sv1.dot(v2), FUZZ);
    assertEquals(dv1.dot(dv2), dv1.dot(v2), FUZZ);

    // first attempt has no cached distances
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), dv1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), sv1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(dv2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(sv2), FUZZ);

    // now repeat with cached sizes
    assertEquals(dv1.getLengthSquared(), v1.getLengthSquared(), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), dv1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), sv1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(dv2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(sv2), FUZZ);

    assertEquals(dv1.minValue(), v1.minValue(), FUZZ);
    assertEquals(dv1.minValueIndex(), v1.minValueIndex());

    assertEquals(dv1.maxValue(), v1.maxValue(), FUZZ);
    assertEquals(dv1.maxValueIndex(), v1.maxValueIndex());

    Vector nv1 = v1.normalize();

    assertEquals(0, dv1.getDistanceSquared(v1), FUZZ);
    assertEquals(1, nv1.norm(2), FUZZ);
    assertEquals(0, dv1.normalize().getDistanceSquared(nv1), FUZZ);

    nv1 = v1.normalize(1);
    assertEquals(0, dv1.getDistanceSquared(v1), FUZZ);
    assertEquals(1, nv1.norm(1), FUZZ);
    assertEquals(0, dv1.normalize(1).getDistanceSquared(nv1), FUZZ);

    assertEquals(dv1.norm(0), v1.norm(0), FUZZ);
    assertEquals(dv1.norm(1), v1.norm(1), FUZZ);
    assertEquals(dv1.norm(1.5), v1.norm(1.5), FUZZ);
    assertEquals(dv1.norm(2), v1.norm(2), FUZZ);

    assertEquals(dv1.zSum(), v1.zSum(), FUZZ);

    assertEquals(3.1 * v1.size(), v1.assign(3.1).zSum(), FUZZ);
    assertEquals(0, v1.plus(-3.1).norm(1), FUZZ);
    v1.assign(dv1);
    assertEquals(0, v1.getDistanceSquared(dv1), FUZZ);

    assertEquals(dv1.zSum() - dv1.size() * 3.4, v1.assign(Functions.minus(3.4)).zSum(), FUZZ);
    assertEquals(dv1.zSum() - dv1.size() * 4.5, v1.assign(Functions.MINUS, 1.1).zSum(), FUZZ);
    v1.assign(dv1);

    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.assign(v2, Functions.MINUS)), FUZZ);
    v1.assign(dv1);

    assertEquals(dv1.norm(2), Math.sqrt(v1.aggregate(Functions.PLUS, Functions.pow(2))), FUZZ);
    assertEquals(dv1.dot(dv2), v1.aggregate(v2, Functions.PLUS, Functions.MULT), FUZZ);

    assertEquals(dv1.viewPart(5, 10).zSum(), v1.viewPart(5, 10).zSum(), FUZZ);

    Vector v3 = v1.clone();

    // must be the right type ... tricky to tell that in the face of type erasure
    assertTrue(v0.getClass().isAssignableFrom(v3.getClass()));
    assertTrue(v3.getClass().isAssignableFrom(v0.getClass()));

    assertEquals(0, v1.getDistanceSquared(v3), FUZZ);
    assertNotSame(v1, v3);
    v3.assign(0);
    assertEquals(0, dv1.getDistanceSquared(v1), FUZZ);
    assertEquals(0, v3.getLengthSquared(), FUZZ);

    dv1.assign(Functions.ABS);
    v1.assign(Functions.ABS);
    assertEquals(0, dv1.logNormalize().getDistanceSquared(v1.logNormalize()), FUZZ);
    assertEquals(0, dv1.logNormalize(1.5).getDistanceSquared(v1.logNormalize(1.5)), FUZZ);

    // aggregate

    // cross,

    // getNumNondefaultElements

    for (Vector.Element element : v1.all()) {
      assertEquals(dv1.get(element.index()), element.get(), 0);
      assertEquals(dv1.get(element.index()), v1.get(element.index()), 0);
      assertEquals(dv1.get(element.index()), v1.getQuick(element.index()), 0);
    }


  }

  abstract Vector generateTestVector(int cardinality);

  Vector getTestVector() {
    return test;
  }

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    test = generateTestVector(2 * values.length + 1);
    for (int i = 0; i < values.length; i++) {
      test.set(2 * i + 1, values[i]);
    }
  }

  @Test
  public void testCardinality() {
    assertEquals("size", 7, test.size());
  }

  @Test
  public void testIterator() {
    Iterator<Vector.Element> iterator = test.nonZeroes().iterator();
    checkIterator(iterator, gold);

    iterator = test.all().iterator();
    checkIterator(iterator, gold);

    double[] doubles = {0.0, 5.0, 0, 3.0};
    RandomAccessSparseVector zeros = new RandomAccessSparseVector(doubles.length);
    for (int i = 0; i < doubles.length; i++) {
      zeros.setQuick(i, doubles[i]);
    }
    iterator = zeros.iterateNonZero();
    checkIterator(iterator, doubles);
    iterator = zeros.iterator();
    checkIterator(iterator, doubles);

    doubles = new double[]{0.0, 0.0, 0, 0.0};
    zeros = new RandomAccessSparseVector(doubles.length);
    for (int i = 0; i < doubles.length; i++) {
      zeros.setQuick(i, doubles[i]);
    }
    iterator = zeros.iterateNonZero();
    checkIterator(iterator, doubles);
    iterator = zeros.iterator();
    checkIterator(iterator, doubles);

  }

  @Test
  public void testIteratorSet() {
    Vector clone = test.clone();
    for (Element e : clone.nonZeroes()) {
      e.set(e.get() * 2.0);
    }
    for (Element e : clone.nonZeroes()) {
      assertEquals(test.get(e.index()) * 2.0, e.get(), EPSILON);
    }
    clone = test.clone();
    for (Element e : clone.all()) {
      e.set(e.get() * 2.0);
    }
    for (Element e : clone.all()) {
      assertEquals(test.get(e.index()) * 2.0, e.get(), EPSILON);
    }
  }

  @Test
  public void testCopy() {
    Vector copy = test.clone();
    for (int i = 0; i < test.size(); i++) {
      assertEquals("copy [" + i + ']', test.get(i), copy.get(i), EPSILON);
    }
  }

  @Test
  public void testGet() {
    for (int i = 0; i < test.size(); i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 0.0, test.get(i), EPSILON);
      } else {
        assertEquals("get [" + i + ']', values[i/2], test.get(i), EPSILON);
      }
    }
  }

  @Test(expected = IndexException.class)
  public void testGetOver() {
    test.get(test.size());
  }

  @Test(expected = IndexException.class)
  public void testGetUnder() {
    test.get(-1);
  }

  @Test
  public void testSet() {
    test.set(3, 4.5);
    for (int i = 0; i < test.size(); i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 0.0, test.get(i), EPSILON);
      } else if (i == 3) {
        assertEquals("set [" + i + ']', 4.5, test.get(i), EPSILON);
      } else {
        assertEquals("set [" + i + ']', values[i/2], test.get(i), EPSILON);
      }
    }
  }

  @Test
  public void testSize() {
    assertEquals("size", 3, test.getNumNondefaultElements());
  }

  @Test
  public void testViewPart() {
    Vector part = test.viewPart(1, 2);
    assertEquals("part size", 2, part.getNumNondefaultElements());
    for (int i = 0; i < part.size(); i++) {
      assertEquals("part[" + i + ']', test.get(i+1), part.get(i), EPSILON);
    }
  }

  @Test(expected = IndexException.class)
  public void testViewPartUnder() {
    test.viewPart(-1, values.length);
  }

  @Test(expected = IndexException.class)
  public void testViewPartOver() {
    test.viewPart(2, 7);
  }

  @Test(expected = IndexException.class)
  public void testViewPartCardinality() {
    test.viewPart(1, 8);
  }

  @Test
  public void testSparseDoubleVectorInt() {
    Vector val = new RandomAccessSparseVector(4);
    assertEquals("size", 4, val.size());
    for (int i = 0; i < 4; i++) {
      assertEquals("get [" + i + ']', 0.0, val.get(i), EPSILON);
    }
  }

  @Test
  public void testDot() {
    double res = test.dot(test);
    double expected = 3.3 * 3.3 + 2.2 * 2.2 + 1.1 * 1.1;
    assertEquals("dot", expected, res, EPSILON);
  }

  @Test
  public void testDot2() {
    Vector test2 = test.clone();
    test2.set(1, 0.0);
    test2.set(3, 0.0);
    assertEquals(3.3 * 3.3, test2.dot(test), EPSILON);
  }

  @Test(expected = CardinalityException.class)
  public void testDotCardinality() {
    test.dot(new DenseVector(test.size() + 1));
  }

  @Test
  public void testNormalize() {
    Vector val = test.normalize();
    double mag = Math.sqrt(1.1 * 1.1 + 2.2 * 2.2 + 3.3 * 3.3);
    for (int i = 0; i < test.size(); i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 0.0, val.get(i), EPSILON);
      } else {
        assertEquals("dot", values[i/2] / mag, val.get(i), EPSILON);
      }
    }
  }

  @Test
  public void testMinus() {
    Vector val = test.minus(test);
    assertEquals("size", test.size(), val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', 0.0, val.get(i), EPSILON);
    }

    val = test.minus(test).minus(test);
    assertEquals("cardinality", test.size(), val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', 0.0, val.get(i) + test.get(i), EPSILON);
    }

    Vector val1 = test.plus(1);
    val = val1.minus(test);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', 1.0, val.get(i), EPSILON);
    }

    val1 = test.plus(-1);
    val = val1.minus(test);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', -1.0, val.get(i), EPSILON);
    }
  }

  @Test
  public void testPlusDouble() {
    Vector val = test.plus(1);
    assertEquals("size", test.size(), val.size());
    for (int i = 0; i < test.size(); i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 1.0, val.get(i), EPSILON);
      } else {
        assertEquals("get [" + i + ']', values[i/2] + 1.0, val.get(i), EPSILON);
      }
    }
  }

  @Test
  public void testPlusVector() {
    Vector val = test.plus(test);
    assertEquals("size", test.size(), val.size());
    for (int i = 0; i < test.size(); i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 0.0, val.get(i), EPSILON);
      } else {
        assertEquals("get [" + i + ']', values[i/2] * 2.0, val.get(i), EPSILON);
      }
    }
  }

  @Test(expected = CardinalityException.class)
  public void testPlusVectorCardinality() {
    test.plus(new DenseVector(test.size() + 1));
  }

  @Test
  public void testTimesDouble() {
    Vector val = test.times(3);
    assertEquals("size", test.size(), val.size());
    for (int i = 0; i < test.size(); i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 0.0, val.get(i), EPSILON);
      } else {
        assertEquals("get [" + i + ']', values[i/2] * 3.0, val.get(i), EPSILON);
      }
    }
  }

  @Test
  public void testDivideDouble() {
    Vector val = test.divide(3);
    assertEquals("size", test.size(), val.size());
    for (int i = 0; i < test.size(); i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 0.0, val.get(i), EPSILON);
      } else {
        assertEquals("get [" + i + ']', values[i/2] / 3.0, val.get(i), EPSILON);
      }
    }
  }

  @Test
  public void testTimesVector() {
    Vector val = test.times(test);
    assertEquals("size", test.size(), val.size());
    for (int i = 0; i < test.size(); i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 0.0, val.get(i), EPSILON);
      } else {
        assertEquals("get [" + i + ']', values[i/2] * values[i/2], val.get(i), EPSILON);
      }
    }
  }

  @Test(expected = CardinalityException.class)
  public void testTimesVectorCardinality() {
    test.times(new DenseVector(test.size() + 1));
  }

  @Test
  public void testZSum() {
    double expected = 0;
    for (double value : values) {
      expected += value;
    }
    assertEquals("wrong zSum", expected, test.zSum(), EPSILON);
  }

  @Test
  public void testGetDistanceSquared() {
    Vector other = new RandomAccessSparseVector(test.size());
    other.set(1, -2);
    other.set(2, -5);
    other.set(3, -9);
    other.set(4, 1);
    double expected = test.minus(other).getLengthSquared();
    assertTrue("a.getDistanceSquared(b) != a.minus(b).getLengthSquared",
               Math.abs(expected - test.getDistanceSquared(other)) < 10.0E-7);
  }

  @Test
  public void testAssignDouble() {
    test.assign(0);
    for (int i = 0; i < values.length; i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i), EPSILON);
    }
  }

  @Test
  public void testAssignDoubleArray() {
    double[] array = new double[test.size()];
    test.assign(array);
    for (int i = 0; i < values.length; i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i), EPSILON);
    }
  }

  @Test(expected = CardinalityException.class)
  public void testAssignDoubleArrayCardinality() {
    double[] array = new double[test.size() + 1];
    test.assign(array);
  }

  @Test
  public void testAssignVector() {
    Vector other = new DenseVector(test.size());
    test.assign(other);
    for (int i = 0; i < values.length; i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i), EPSILON);
    }
  }

  @Test(expected = CardinalityException.class)
  public void testAssignVectorCardinality() {
    Vector other = new DenseVector(test.size() - 1);
    test.assign(other);
  }

  @Test
  public void testAssignUnaryFunction() {
    test.assign(Functions.NEGATE);
    for (int i = 1; i < values.length; i += 2) {
      assertEquals("value[" + i + ']', -values[i], test.getQuick(i+2), EPSILON);
    }
  }

  @Test
  public void testAssignBinaryFunction() {
    test.assign(test, Functions.PLUS);
    for (int i = 0; i < values.length; i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 0.0, test.get(i), EPSILON);
      } else {
        assertEquals("value[" + i + ']', 2 * values[i - 1], test.getQuick(i), EPSILON);
      }
    }
  }

  @Test
  public void testAssignBinaryFunction2() {
    test.assign(Functions.plus(4));
    for (int i = 0; i < values.length; i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 4.0, test.get(i), EPSILON);
      } else {
        assertEquals("value[" + i + ']', values[i - 1] + 4, test.getQuick(i), EPSILON);
      }
    }
  }

  @Test
  public void testAssignBinaryFunction3() {
    test.assign(Functions.mult(4));
    for (int i = 0; i < values.length; i++) {
      if (i % 2 == 0) {
        assertEquals("get [" + i + ']', 0.0, test.get(i), EPSILON);
      } else {
        assertEquals("value[" + i + ']', values[i - 1] * 4, test.getQuick(i), EPSILON);
      }
    }
  }

  @Test
  public void testLike() {
    Vector other = test.like();
    assertTrue("not like", test.getClass().isAssignableFrom(other.getClass()));
    assertEquals("size", test.size(), other.size());
  }

  @Test
  public void testCrossProduct() {
    Matrix result = test.cross(test);
    assertEquals("row size", test.size(), result.rowSize());
    assertEquals("col size", test.size(), result.columnSize());
    for (int row = 0; row < result.rowSize(); row++) {
      for (int col = 0; col < result.columnSize(); col++) {
        assertEquals("cross[" + row + "][" + col + ']', test.getQuick(row)
            * test.getQuick(col), result.getQuick(row, col), EPSILON);

      }
    }
  }

  @Test
  public void testIterators() {
    final T v0 = vectorToTest(20);

    double sum = 0;
    int elements = 0;
    int nonZero = 0;
    for (Element element : v0.all()) {
      elements++;
      sum += element.get();
      if (element.get() != 0) {
        nonZero++;
      }
    }

    int nonZeroIterated = Iterables.size(v0.nonZeroes());
    assertEquals(20, elements);
    assertEquals(v0.size(), elements);
    assertEquals(nonZeroIterated, nonZero);
    assertEquals(v0.zSum(), sum, 0);
  }

  @Test
  public void testSmallDistances() {
    for (double fuzz : new double[]{1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10}) {
      MultiNormal x = new MultiNormal(fuzz, new ConstantVector(0, 20));
      for (int i = 0; i < 10000; i++) {
        final T v1 = vectorToTest(20);
        Vector v2 = v1.plus(x.sample());
        if (1 + fuzz * fuzz > 1) {
          String msg = String.format("fuzz = %.1g, >", fuzz);
          assertTrue(msg, v1.getDistanceSquared(v2) > 0);
          assertTrue(msg, v2.getDistanceSquared(v1) > 0);
        } else {
          String msg = String.format("fuzz = %.1g, >=", fuzz);
          assertTrue(msg, v1.getDistanceSquared(v2) >= 0);
          assertTrue(msg, v2.getDistanceSquared(v1) >= 0);
        }
      }
    }
  }


  public void testToString() {
    Vector w;

    w = generateTestVector(20);
    w.set(0, 1.1);
    w.set(13, 100500.);
    w.set(19, 3.141592);
    assertEquals("{0:1.1,13:100500.0,19:3.141592}", w.toString());

    w = generateTestVector(12);
    w.set(10, 0.1);
    assertEquals("{10:0.1}", w.toString());

    w = generateTestVector(12);
    assertEquals("{}", w.toString());
  }
}
