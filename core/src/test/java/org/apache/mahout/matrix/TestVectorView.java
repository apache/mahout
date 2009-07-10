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

import java.util.Iterator;

public class TestVectorView extends TestCase {

  private static final int cardinality = 3;

  private static final int offset = 1;

  final double[] values = {0.0, 1.1, 2.2, 3.3, 4.4, 5.5};

  final Vector test = new VectorView(new DenseVector(values), offset,
      cardinality);

  public TestVectorView(String name) {
    super(name);
  }

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
      assertTrue(true);
    }
  }

  public void testIterator() throws Exception {

    VectorView view = new VectorView(new DenseVector(values), offset, cardinality);
    double[] gold = {1.1, 2.2, 3.3};
    Iterator<Vector.Element> iter = view.iterateAll();
    checkIterator(iter, gold);
    iter = view.iterateNonZero();
    checkIterator(iter, gold);

    view = new VectorView(new DenseVector(values), 0, cardinality);
    gold = new double[]{0.0, 1.1, 2.2};
    iter = view.iterateAll();
    checkIterator(iter, gold);
    gold = new double[]{1.1, 2.2};
    iter = view.iterateNonZero();
    checkIterator(iter, gold);

  }

  private static void checkIterator(Iterator<Vector.Element> iter, double[] gold) {
    int i = 0;
    while (iter.hasNext()) {
      Vector.Element elt = iter.next();
      assertTrue((elt.index()) + " Value: " + gold[i]
          + " does not equal: " + elt.get(), gold[i] == elt.get());
      i++;
    }
  }

  public void testGetUnder() {
    try {
      test.get(-1);
      fail("expected exception");
    } catch (IndexException e) {
      assertTrue(true);
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
    } catch (CardinalityException e) {
      fail("expected index exception");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testViewPartOver() {
    try {
      test.viewPart(2, cardinality);
      fail("no exception");
    } catch (CardinalityException e) {
      fail("expected index exception");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testViewPartCardinality() {
    try {
      test.viewPart(1, values.length + 1);
      fail("no exception");
    } catch (CardinalityException e) {
      assertTrue(true);
    } catch (IndexException e) {
      fail("expected cardinality exception");
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
      assertTrue(true);
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
      assertTrue(true);
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
      assertTrue(true);
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
      assertTrue(true);
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
      assertTrue(true);
    }
  }

  public void testAssignUnaryFunction() {
    test.assign(new NegateFunction());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', -values[i + 1], test.getQuick(i));
    }
  }

  public void testAssignBinaryFunction() throws Exception {
    test.assign(test, new PlusFunction());
    for (int i = 0; i < test.size(); i++) {
      assertEquals("value[" + i + ']', 2 * values[i + 1], test.getQuick(i));
    }
  }

  public void testAssignBinaryFunction2() throws Exception {
    test.assign(new PlusFunction(), 4);
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

  public void testAssignBinaryFunctionCardinality() {
    try {
      test.assign(test.like(2), new PlusFunction());
      fail("Cardinality exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testThisHaveSharedCells() throws Exception {
    assertTrue("test not shared?", test.haveSharedCells(test));
  }

  public void testViewHaveSharedCells() throws Exception {
    Vector view = test.viewPart(1, 2);
    assertTrue("view not shared?", view.haveSharedCells(test));
    assertTrue("test not shared?", test.haveSharedCells(view));
  }

  public void testViewsHaveSharedCells() throws Exception {
    Vector view1 = test.viewPart(0, 2);
    Vector view2 = test.viewPart(1, 2);
    assertTrue("view1 not shared?", view1.haveSharedCells(view2));
    assertTrue("view2 not shared?", view2.haveSharedCells(view1));
  }

  public void testLike() {
    assertTrue("not like", test.like() instanceof DenseVector);
  }

  public void testLikeN() {
    Vector other = test.like(5);
    assertTrue("not like", other instanceof DenseVector);
    assertEquals("size", 5, other.size());
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
