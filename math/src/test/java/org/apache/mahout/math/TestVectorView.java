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

import org.apache.mahout.math.function.TimesFunction;

import static org.apache.mahout.math.function.Functions.*;

import java.util.Iterator;

public class TestVectorView extends MahoutTestCase {

  private static final int cardinality = 3;

  private static final int offset = 1;

  private final double[] values = {0.0, 1.1, 2.2, 3.3, 4.4, 5.5};

  private final Vector test = new VectorView(new DenseVector(values), offset,
      cardinality);

  public void testAsFormatString() {
    String formatString = test.asFormatString();
    Vector v = AbstractVector.decodeVector(formatString);
    assertEquals("size", test.size(), v.size());
  }

  public void testCardinality() {
    assertEquals("size", 3, test.size());
  }

  public void testCopy() throws Exception {
    Vector copy = test.clone();
    for (int i = 0; i < test.size(); i++) {
      assertEquals("copy [" + i + ']', test.get(i), copy.get(i));
    }
  }

  public void testGet() throws Exception {
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[i + offset], test.get(i));
    }
  }

  public void testGetOver() {
    try {
      test.get(test.size());
      fail("expected exception");
    } catch (IndexException e) {
    }
  }

  public void testIterator() throws Exception {

    VectorView view = new VectorView(new DenseVector(values), offset, cardinality);
    double[] gold = {1.1, 2.2, 3.3};
    Iterator<Vector.Element> iter = view.iterator();
    checkIterator(iter, gold);
    iter = view.iterateNonZero();
    checkIterator(iter, gold);

    view = new VectorView(new DenseVector(values), 0, cardinality);
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
      assertEquals((elt.index()) + " Value: " + gold[i]
          + " does not equal: " + elt.get(), gold[i], elt.get(), 0.0);
      i++;
    }
  }

  public void testGetUnder() {
    try {
      test.get(-1);
      fail("expected exception");
    } catch (IndexException e) {
    }
  }

  public void testSet() throws Exception {
    test.set(2, 4.5);
    for (int i = 0; i < test.size(); i++) {
      if (i == 2) {
        assertEquals("set [" + i + ']', 4.5, test.get(i));
      } else {
        assertEquals("set [" + i + ']', values[offset + i], test.get(i));
      }
    }
  }

  public void testSize() throws Exception {
    assertEquals("size", 3, test.getNumNondefaultElements());
  }

  public void testViewPart() throws Exception {
    Vector part = test.viewPart(1, 2);
    assertEquals("part size", 2, part.getNumNondefaultElements());
    for (int i = 0; i < part.size(); i++) {
      assertEquals("part[" + i + ']', values[offset + i + 1], part.get(i));
    }
  }

  public void testViewPartUnder() {
    try {
      test.viewPart(-1, cardinality);
      fail("no exception");
    } catch (IndexException e) {
    }
  }

  public void testViewPartOver() {
    try {
      test.viewPart(2, cardinality);
      fail("no exception");
    } catch (IndexException e) {
    }
  }

  public void testViewPartCardinality() {
    try {
      test.viewPart(1, values.length + 1);
      fail("no exception");
    } catch (IndexException e) {
    }
  }

  public void testDot() throws Exception {
    double res = test.dot(test);
    assertEquals("dot", 1.1 * 1.1 + 2.2 * 2.2 + 3.3 * 3.3, res);
  }

  public void testDotCardinality() {
    try {
      test.dot(new DenseVector(test.size() + 1));
      fail("expected exception");
    } catch (CardinalityException e) {
    }
  }

  public void testNormalize() throws Exception {
    Vector res = test.normalize();
    double mag = Math.sqrt(1.1 * 1.1 + 2.2 * 2.2 + 3.3 * 3.3);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("dot", values[offset + i] / mag, res.get(i));
    }
  }

  public void testMinus() throws Exception {
    Vector val = test.minus(test);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', 0.0, val.get(i));
    }
  }

  public void testPlusDouble() throws Exception {
    Vector val = test.plus(1);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[offset + i] + 1, val.get(i));
    }
  }

  public void testPlusVector() throws Exception {
    Vector val = test.plus(test);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[offset + i] * 2, val.get(i));
    }
  }

  public void testPlusVectorCardinality() {
    try {
      test.plus(new DenseVector(test.size() + 1));
      fail("expected exception");
    } catch (CardinalityException e) {
    }
  }

  public void testTimesDouble() throws Exception {
    Vector val = test.times(3);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[offset + i] * 3, val.get(i));
    }
  }

  public void testDivideDouble() throws Exception {
    Vector val = test.divide(3);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[offset + i] / 3, val.get(i));
    }
  }

  public void testTimesVector() throws Exception {
    Vector val = test.times(test);
    assertEquals("size", 3, val.size());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("get [" + i + ']', values[offset + i] * values[offset + i],
          val.get(i));
    }
  }

  public void testTimesVectorCardinality() {
    try {
      test.times(new DenseVector(test.size() + 1));
      fail("expected exception");
    } catch (CardinalityException e) {
    }
  }

  public void testZSum() {
    double expected = 0;
    for (int i = offset; i < offset + cardinality; i++) {
      expected += values[i];
    }
    assertEquals("wrong zSum", expected, test.zSum());
  }

  public void testAssignDouble() {
    test.assign(0);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i));
    }
  }

  public void testAssignDoubleArray() throws Exception {
    double[] array = new double[test.size()];
    test.assign(array);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i));
    }
  }

  public void testAssignDoubleArrayCardinality() {
    double[] array = new double[test.size() + 1];
    try {
      test.assign(array);
      fail("cardinality exception expected");
    } catch (CardinalityException e) {
    }
  }

  public void testAssignVector() throws Exception {
    Vector other = new DenseVector(test.size());
    test.assign(other);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i));
    }
  }

  public void testAssignVectorCardinality() {
    Vector other = new DenseVector(test.size() - 1);
    try {
      test.assign(other);
      fail("cardinality exception expected");
    } catch (CardinalityException e) {
    }
  }

  public void testAssignUnaryFunction() {
    test.assign(negate);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', -values[i + 1], test.getQuick(i));
    }
  }

  public void testAssignBinaryFunction() throws Exception {
    test.assign(test, plus);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', 2 * values[i + 1], test.getQuick(i));
    }
  }

  public void testAssignBinaryFunction2() throws Exception {
    test.assign(plus, 4);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', values[i + 1] + 4, test.getQuick(i));
    }
  }

  public void testAssignBinaryFunction3() throws Exception {
    test.assign(new TimesFunction(), 4);
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', values[i + 1] * 4, test.getQuick(i));
    }
  }

  public void testLike() {
    assertTrue("not like", test.like() instanceof VectorView);
  }

  public void testCrossProduct() {
    Matrix result = test.cross(test);
    assertEquals("row size", test.size(), result.size()[0]);
    assertEquals("col size", test.size(), result.size()[1]);
    for (int row = 0; row < result.size()[0]; row++) {
      for (int col = 0; col < result.size()[1]; col++) {
        assertEquals("cross[" + row + "][" + col + ']', test.getQuick(row)
            * test.getQuick(col), result.getQuick(row, col));
      }
    }
  }
}
