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

public class TestDenseVector extends TestCase {

  final double[] values = { 1.1, 2.2, 3.3 };

  final Vector test = new DenseVector(values);

  public TestDenseVector(String name) {
    super(name);
  }

  public void testAsFormatString() {
    String formatString = test.asWritableComparable().toString();
    assertEquals("format", "[, 1.1, 2.2, 3.3, ] ", formatString);
  }

  public void testCardinality() {
    assertEquals("cardinality", 3, test.cardinality());
  }

  public void testCopy() throws Exception {
    Vector copy = test.copy();
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("copy [" + i + ']', test.get(i), copy.get(i));
  }

  public void testGet() throws Exception {
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', values[i], test.get(i));
  }

  public void testGetOver() {
    try {
      test.get(test.cardinality());
      fail("expected exception");
    } catch (IndexException e) {
      assertTrue(true);
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
    for (int i = 0; i < test.cardinality(); i++)
      if (i == 2)
        assertEquals("set [" + i + ']', 4.5, test.get(i));
      else
        assertEquals("set [" + i + ']', values[i], test.get(i));
  }

  public void testSize() throws Exception {
    assertEquals("size", 3, test.size());
  }

  public void testToArray() throws Exception {
    double[] val = test.toArray();
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', val[i], test.get(i));
  }

  public void testViewPart() throws Exception {
    Vector part = test.viewPart(1, 2);
    assertEquals("part size", 2, part.size());
    for (int i = 0; i < part.cardinality(); i++)
      assertEquals("part[" + i + ']', values[i + 1], part.get(i));
  }

  public void testViewPartUnder() {
    try {
      test.viewPart(-1, values.length);
      fail("no exception");
    } catch (CardinalityException e) {
      fail("wrong exception");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testViewPartOver() {
    try {
      test.viewPart(2, values.length);
      fail("no exception");
    } catch (CardinalityException e) {
      fail("wrong exception");
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
      fail("wrong exception");
    }
  }

  public void testDecodeFormat() throws Exception {
    Vector val = DenseVector.decodeFormat(test.asWritableComparable());
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', test.get(i), val.get(i));
  }

  public void testDenseVectorDoubleArray() throws Exception {
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("test[" + i + ']', values[i], test.get(i));
  }

  public void testDenseVectorInt() throws Exception {
    Vector val = new DenseVector(4);
    assertEquals("cardinality", 4, val.cardinality());
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', 0.0, val.get(i));
  }

  public void testDot() throws Exception {
    double res = test.dot(test);
    assertEquals("dot", 1.1 * 1.1 + 2.2 * 2.2 + 3.3 * 3.3, res);
  }

  public void testDotCardinality() {
    try {
      test.dot(new DenseVector(test.cardinality() + 1));
      fail("expected exception");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testNormalize() throws Exception {
    Vector res = test.normalize();
    double mag = Math.sqrt(1.1 * 1.1 + 2.2 * 2.2 + 3.3 * 3.3);
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("dot", values[i] / mag, res.get(i));
  }

  public void testMinus() throws Exception {
    Vector val = test.minus(test);
    assertEquals("cardinality", 3, val.cardinality());
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', 0.0, val.get(i));
  }

  public void testPlusDouble() throws Exception {
    Vector val = test.plus(1);
    assertEquals("cardinality", 3, val.cardinality());
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', values[i] + 1, val.get(i));
  }

  public void testPlusVector() throws Exception {
    Vector val = test.plus(test);
    assertEquals("cardinality", 3, val.cardinality());
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', values[i] * 2, val.get(i));
  }

  public void testPlusVectorCardinality() {
    try {
      test.plus(new DenseVector(test.cardinality() + 1));
      fail("expected exception");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testTimesDouble() throws Exception {
    Vector val = test.times(3);
    assertEquals("cardinality", 3, val.cardinality());
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', values[i] * 3, val.get(i));
  }

  public void testDivideDouble() throws Exception {
    Vector val = test.divide(3);
    assertEquals("cardinality", 3, val.cardinality());
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', values[i] / 3, val.get(i));
  }

  public void testTimesVector() throws Exception {
    Vector val = test.times(test);
    assertEquals("cardinality", 3, val.cardinality());
    for (int i = 0; i < test.cardinality(); i++)
      assertEquals("get [" + i + ']', values[i] * values[i], val.get(i));
  }

  public void testTimesVectorCardinality() {
    try {
      test.times(new DenseVector(test.cardinality() + 1));
      fail("expected exception");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testZSum() {
    double expected = 0;
    for (double value : values) {
      expected += value;
    }
    assertEquals("wrong zSum", expected, test.zSum());
  }

  public void testAssignDouble() {
    test.assign(0);
    for (int i = 0; i < values.length; i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i));
    }
  }

  public void testAssignDoubleArray() throws Exception {
    double[] array = new double[test.cardinality()];
    test.assign(array);
    for (int i = 0; i < values.length; i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i));
    }
  }

  public void testAssignDoubleArrayCardinality() {
    double[] array = new double[test.cardinality() + 1];
    try {
      test.assign(array);
      fail("cardinality exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testAssignVector() throws Exception {
    Vector other = new DenseVector(test.cardinality());
    test.assign(other);
    for (int i = 0; i < values.length; i++) {
      assertEquals("value[" + i + ']', 0.0, test.getQuick(i));
    }
  }

  public void testAssignVectorCardinality() {
    Vector other = new DenseVector(test.cardinality() - 1);
    try {
      test.assign(other);
      fail("cardinality exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testAssignUnaryFunction() {
    test.assign(new NegateFunction());
    for (int i = 0; i < values.length; i++) {
      assertEquals("value[" + i + ']', -values[i], test.getQuick(i));
    }
  }

  public void testAssignBinaryFunction() throws Exception {
    test.assign(test, new PlusFunction());
    for (int i = 0; i < values.length; i++) {
      assertEquals("value[" + i + ']', 2 * values[i], test.getQuick(i));
    }
  }

  public void testAssignBinaryFunction2() throws Exception {
    test.assign(new PlusFunction(), 4);
    for (int i = 0; i < values.length; i++) {
      assertEquals("value[" + i + ']', values[i] + 4, test.getQuick(i));
    }
  }

  public void testAssignBinaryFunction3() throws Exception {
    test.assign(new TimesFunction(), 4);
    for (int i = 0; i < values.length; i++)
      assertEquals("value[" + i + ']', values[i] * 4, test.getQuick(i));
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
    assertEquals("cardinality", 5, other.cardinality());
  }

  public void testCrossProduct() {
    Matrix result = test.cross(test);
    assertEquals("row cardinality", test.cardinality(), result.cardinality()[0]);
    assertEquals("col cardinality", test.cardinality(), result.cardinality()[1]);
    for (int row = 0; row < result.cardinality()[0]; row++) {
      for (int col = 0; col < result.cardinality()[1]; col++) {
        assertEquals("cross[" + row + "][" + col + ']', test.getQuick(row)
            * test.getQuick(col), result.getQuick(row, col));
      }
    }
  }
}
