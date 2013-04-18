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

import java.util.Iterator;

import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.TimesFunction;
import org.junit.Test;

public final class TestVectorView extends MahoutTestCase {

  private static final int CARDINALITY = 3;

  private static final int OFFSET = 1;

  private final double[] values = {0.0, 1.1, 2.2, 3.3, 4.4, 5.5};

  private final Vector test = new VectorView(new DenseVector(values), OFFSET, CARDINALITY);

  @Test
  public void testCardinality() {
    assertEquals("size", 3, test.size());
  }

  @Test
  public void testCopy() throws Exception {
    Vector copy = test.clone();
    for (int i = 0; i < test.size(); i++) {
      assertEquals("copy [" + i + ']', test.get(i), copy.get(i), EPSILON);
    }
  }

  @Test
  public void testGet() throws Exception {
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[i + OFFSET], test.get(i), EPSILON);
    }
  }

  @Test(expected = IndexException.class)
  public void testGetOver() {
    test.get(test.size());
  }

  @Test
  public void testIterator() throws Exception {

    VectorView view = new VectorView(new DenseVector(values), OFFSET, CARDINALITY);
    double[] gold = {1.1, 2.2, 3.3};
    Iterator<Vector.Element> iter = view.iterator();
    checkIterator(iter, gold);
    iter = view.iterateNonZero();
    checkIterator(iter, gold);

    view = new VectorView(new DenseVector(values), 0, CARDINALITY);
    gold = new double[]{0.0, 1.1, 2.2};
    iter = view.iterator();
    checkIterator(iter, gold);
    gold = new double[]{1.1, 2.2};
    iter = view.iterateNonZero();
    checkIterator(iter, gold);

  }

  private static void checkIterator(Iterator<Vector.Element> iter, double[] gold) {
    int i = 0;
    while (iter.hasNext()) {
      Vector.Element elt = iter.next();
      assertEquals(elt.index() + " Value: " + gold[i]
          + " does not equal: " + elt.get(), gold[i], elt.get(), 0.0);
      i++;
    }
  }

  @Test(expected = IndexException.class)
  public void testGetUnder() {
    test.get(-1);
  }

  @Test
  public void testSet() throws Exception {
    test.set(2, 4.5);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("set [" + i + ']', i == 2 ? 4.5 : values[OFFSET + i], test.get(i), EPSILON);
    }
  }

  @Test
  public void testSize() throws Exception {
    assertEquals("size", 3, test.getNumNondefaultElements());
  }

  @Test
  public void testViewPart() throws Exception {
    Vector part = test.viewPart(1, 2);
    assertEquals("part size", 2, part.getNumNondefaultElements());
    for (int i = 0; i < part.size(); i++) {
      assertEquals("part[" + i + ']', values[OFFSET + i + 1], part.get(i), EPSILON);
    }
  }

  @Test(expected = IndexException.class)
  public void testViewPartUnder() {
    test.viewPart(-1, CARDINALITY);
  }

  @Test(expected = IndexException.class)
  public void testViewPartOver() {
    test.viewPart(2, CARDINALITY);
  }

  @Test(expected = IndexException.class)
  public void testViewPartCardinality() {
    test.viewPart(1, values.length + 1);
  }

  @Test
  public void testDot() throws Exception {
    double res = test.dot(test);
    assertEquals("dot", 1.1 * 1.1 + 2.2 * 2.2 + 3.3 * 3.3, res, EPSILON);
  }

  @Test(expected = CardinalityException.class)
  public void testDotCardinality() {
    test.dot(new DenseVector(test.size() + 1));
  }

  @Test
  public void testNormalize() throws Exception {
    Vector res = test.normalize();
    double mag = Math.sqrt(1.1 * 1.1 + 2.2 * 2.2 + 3.3 * 3.3);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("dot", values[OFFSET + i] / mag, res.get(i), EPSILON);
    }
  }

  @Test
  public void testMinus() throws Exception {
    Vector val = test.minus(test);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', 0.0, val.get(i), EPSILON);
    }
  }

  @Test
  public void testPlusDouble() throws Exception {
    Vector val = test.plus(1);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[OFFSET + i] + 1, val.get(i), EPSILON);
    }
  }

  @Test
  public void testPlusVector() throws Exception {
    Vector val = test.plus(test);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[OFFSET + i] * 2, val.get(i), EPSILON);
    }
  }

  @Test(expected = CardinalityException.class)
  public void testPlusVectorCardinality() {
    test.plus(new DenseVector(test.size() + 1));
  }

  @Test
  public void testTimesDouble() throws Exception {
    Vector val = test.times(3);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[OFFSET + i] * 3, val.get(i), EPSILON);
    }
  }

  @Test
  public void testDivideDouble() throws Exception {
    Vector val = test.divide(3);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[OFFSET + i] / 3, val.get(i), EPSILON);
    }
  }

  @Test
  public void testTimesVector() throws Exception {
    Vector val = test.times(test);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[OFFSET + i] * values[OFFSET + i],
          val.get(i), EPSILON);
    }
  }

  @Test(expected = CardinalityException.class)
  public void testTimesVectorCardinality() {
    test.times(new DenseVector(test.size() + 1));
  }

  @Test
  public void testZSum() {
    double expected = 0;
    for (int i = OFFSET; i < OFFSET + CARDINALITY; i++) {
      expected += values[i];
    }
    assertEquals("wrong zSum", expected, test.zSum(), EPSILON);
  }

  @Test
  public void testAssignDouble() {
    test.assign(0);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i), EPSILON);
    }
  }

  @Test
  public void testAssignDoubleArray() throws Exception {
    double[] array = new double[test.size()];
    test.assign(array);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i), EPSILON);
    }
  }

  @Test(expected = CardinalityException.class)
  public void testAssignDoubleArrayCardinality() {
    double[] array = new double[test.size() + 1];
    test.assign(array);
  }

  @Test
  public void testAssignVector() throws Exception {
    Vector other = new DenseVector(test.size());
    test.assign(other);
    for (int i = 0; i < test.size(); i++) {
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
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', -values[i + 1], test.getQuick(i), EPSILON);
    }
  }

  @Test
  public void testAssignBinaryFunction() throws Exception {
    test.assign(test, Functions.PLUS);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', 2 * values[i + 1], test.getQuick(i), EPSILON);
    }
  }

  @Test
  public void testAssignBinaryFunction2() throws Exception {
    test.assign(Functions.PLUS, 4);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', values[i + 1] + 4, test.getQuick(i), EPSILON);
    }
  }

  @Test
  public void testAssignBinaryFunction3() throws Exception {
    test.assign(new TimesFunction(), 4);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', values[i + 1] * 4, test.getQuick(i), EPSILON);
    }
  }

  @Test
  public void testLike() {
    assertTrue("not like", test.like() instanceof VectorView);
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
