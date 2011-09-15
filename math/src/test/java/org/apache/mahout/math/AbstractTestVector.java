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

import org.apache.mahout.math.function.Functions;
import org.junit.Before;
import org.junit.Test;

import java.util.Iterator;

public abstract class AbstractTestVector extends MahoutTestCase {

  private static final double[] values = {1.1, 2.2, 3.3};
  private static final double[] gold = {0.0, 1.1, 0.0, 2.2, 0.0, 3.3, 0.0};

  private Vector test;

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
      test.set(2*i + 1, values[i]);
    }
  }

  @Test
  public void testCardinality() {
    assertEquals("size", 7, test.size());
  }

  @Test
  public void testIterator() {
    Iterator<Vector.Element> iterator = test.iterateNonZero();
    checkIterator(iterator, gold);

    iterator = test.iterator();
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

  private static void checkIterator(Iterator<Vector.Element> nzIter, double[] values) {
    while (nzIter.hasNext()) {
      Vector.Element elt = nzIter.next();
      assertEquals(elt.index() + " Value: " + values[elt.index()]
          + " does not equal: " + elt.get(), values[elt.index()], elt.get(), 0.0);
    }
  }

  @Test
  public void testIteratorSet() {
    Vector clone = test.clone();
    Iterator<Vector.Element> it = clone.iterateNonZero();
    while (it.hasNext()) {
      Vector.Element e = it.next();
      e.set(e.get() * 2.0);
    }
    it = clone.iterateNonZero();
    while (it.hasNext()) {
      Vector.Element e = it.next();
      assertEquals(test.get(e.index()) * 2.0, e.get(), EPSILON);
    }
    clone = test.clone();
    it = clone.iterator();
    while (it.hasNext()) {
      Vector.Element e = it.next();
      e.set(e.get() * 2.0);
    }
    it = clone.iterator();
    while (it.hasNext()) {
      Vector.Element e = it.next();
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

}
