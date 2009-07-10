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
import org.apache.hadoop.io.DataOutputBuffer;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public abstract class MatrixTest extends TestCase {

  protected static final int ROW = AbstractMatrix.ROW;

  protected static final int COL = AbstractMatrix.COL;

  protected final double[][] values = {{1.1, 2.2}, {3.3, 4.4},
      {5.5, 6.6}};

  protected Matrix test;

  protected MatrixTest(String name) {
    super(name);
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    test = matrixFactory(values);
  }

  public abstract Matrix matrixFactory(double[][] values);

  public void testCardinality() {
    int[] c = test.size();
    assertEquals("row cardinality", values.length, c[ROW]);
    assertEquals("col cardinality", values[0].length, c[COL]);
  }

  public void testCopy() {
    int[] c = test.size();
    Matrix copy = test.clone();
    assertEquals("wrong class", copy.getClass(), test.getClass());
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            test.getQuick(row, col), copy.getQuick(row, col));
      }
    }
  }

  public void testGetQuick() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', values[row][col], test
            .getQuick(row, col));
      }
    }
  }

  public void testHaveSharedCells() {
    assertTrue("same", test.haveSharedCells(test));
    assertFalse("different", test.haveSharedCells(test.clone()));
  }

  public void testLike() {
    Matrix like = test.like();
    assertEquals("type", like.getClass(), test.getClass());
    assertEquals("rows", test.size()[ROW], like.size()[ROW]);
    assertEquals("columns", test.size()[COL], like.size()[COL]);
  }

  public void testLikeIntInt() {
    Matrix like = test.like(4, 4);
    assertEquals("type", like.getClass(), test.getClass());
    assertEquals("rows", 4, like.size()[ROW]);
    assertEquals("columns", 4, like.size()[COL]);
  }

  public void testSetQuick() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        test.setQuick(row, col, 1.23);
        assertEquals("value[" + row + "][" + col + ']', 1.23, test.getQuick(
            row, col));
      }
    }
  }

  public void testSize() {
    int[] c = test.getNumNondefaultElements();
    assertEquals("row size", values.length, c[ROW]);
    assertEquals("col size", values[0].length, c[COL]);
  }

  public void testViewPart() {
    int[] offset = {1, 1};
    int[] size = {2, 1};
    Matrix view = test.viewPart(offset, size);
    int[] c = view.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1], view.getQuick(row, col));
      }
    }
  }

  public void testViewPartCardinality() {
    int[] offset = {1, 1};
    int[] size = {3, 3};
    try {
      test.viewPart(offset, size);
      fail("exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    } catch (IndexException e) {
      fail("cardinality exception expected");
    }
  }

  public void testViewPartIndexOver() {
    int[] offset = {1, 1};
    int[] size = {2, 2};
    try {
      test.viewPart(offset, size);
      fail("exception expected");
    } catch (CardinalityException e) {
      fail("index exception expected");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testViewPartIndexUnder() {
    int[] offset = {-1, -1};
    int[] size = {2, 2};
    try {
      test.viewPart(offset, size);
      fail("exception expected");
    } catch (CardinalityException e) {
      fail("index exception expected");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testAssignDouble() {
    int[] c = test.size();
    test.assign(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', 4.53, test.getQuick(
            row, col));
      }
    }
  }

  public void testAssignDoubleArrayArray() {
    int[] c = test.size();
    test.assign(new double[3][2]);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', 0.0, test.getQuick(row,
            col));
      }
    }
  }

  public void testAssignDoubleArrayArrayCardinality() {
    int[] c = test.size();
    try {
      test.assign(new double[c[ROW] + 1][c[COL]]);
      fail("exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testAssignMatrixBinaryFunction() {
    int[] c = test.size();
    test.assign(test, new PlusFunction());
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', 2 * values[row][col],
            test.getQuick(row, col));
      }
    }
  }

  public void testAssignMatrixBinaryFunctionCardinality() {
    try {
      test.assign(test.transpose(), new PlusFunction());
      fail("exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testAssignMatrix() {
    int[] c = test.size();
    Matrix value = test.like();
    value.assign(test);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            test.getQuick(row, col), value.getQuick(row, col));
      }
    }
  }

  public void testAssignMatrixCardinality() {
    try {
      test.assign(test.transpose());
      fail("exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testAssignUnaryFunction() {
    int[] c = test.size();
    test.assign(new NegateFunction());
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', -values[row][col], test
            .getQuick(row, col));
      }
    }
  }

  public void testDivide() {
    int[] c = test.size();
    Matrix value = test.divide(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row][col] / 4.53, value.getQuick(row, col));
      }
    }
  }

  public void testGet() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', values[row][col], test
            .get(row, col));
      }
    }
  }

  public void testGetIndexUnder() {
    int[] c = test.size();
    try {
      for (int row = -1; row < c[ROW]; row++) {
        for (int col = 0; col < c[COL]; col++) {
          test.get(row, col);
        }
      }
      fail("index exception expected");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testGetIndexOver() {
    int[] c = test.size();
    try {
      for (int row = 0; row < c[ROW] + 1; row++) {
        for (int col = 0; col < c[COL]; col++) {
          test.get(row, col);
        }
      }
      fail("index exception expected");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testMinus() {
    int[] c = test.size();
    Matrix value = test.minus(test);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', 0.0, value.getQuick(
            row, col));
      }
    }
  }

  public void testMinusCardinality() {
    try {
      test.minus(test.transpose());
      fail("cardinality exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testPlusDouble() {
    int[] c = test.size();
    Matrix value = test.plus(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row][col] + 4.53, value.getQuick(row, col));
      }
    }
  }

  public void testPlusMatrix() {
    int[] c = test.size();
    Matrix value = test.plus(test);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', values[row][col] * 2,
            value.getQuick(row, col));
      }
    }
  }

  public void testPlusMatrixCardinality() {
    try {
      test.plus(test.transpose());
      fail("cardinality exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testSetUnder() {
    int[] c = test.size();
    try {
      for (int row = -1; row < c[ROW]; row++) {
        for (int col = 0; col < c[COL]; col++) {
          test.set(row, col, 1.23);
        }
      }
      fail("index exception expected");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testSetOver() {
    int[] c = test.size();
    try {
      for (int row = 0; row < c[ROW] + 1; row++) {
        for (int col = 0; col < c[COL]; col++) {
          test.set(row, col, 1.23);
        }
      }
      fail("index exception expected");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testTimesDouble() {
    int[] c = test.size();
    Matrix value = test.times(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row][col] * 4.53, value.getQuick(row, col));
      }
    }
  }

  public void testTimesMatrix() {
    int[] c = test.size();
    Matrix transpose = test.transpose();
    Matrix value = test.times(transpose);
    int[] v = value.size();
    assertEquals("rows", c[ROW], v[ROW]);
    assertEquals("cols", c[ROW], v[COL]);
    // TODO: check the math too, lazy
    Matrix timestest = new DenseMatrix(10, 1);
    /* will throw ArrayIndexOutOfBoundsException exception without MAHOUT-26 */
    timestest.transpose().times(timestest);
  }

  public void testTimesMatrixCardinality() {
    Matrix other = test.like(5, 8);
    try {
      test.times(other);
      fail("cardinality exception expected");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testTranspose() {
    int[] c = test.size();
    Matrix transpose = test.transpose();
    int[] t = transpose.size();
    assertEquals("rows", c[COL], t[ROW]);
    assertEquals("cols", c[ROW], t[COL]);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            test.getQuick(row, col), transpose.getQuick(col, row));
      }
    }
  }

  public void testZSum() {
    double sum = test.zSum();
    assertEquals("zsum", 23.1, sum);
  }

  public void testAssignRow() {
    double[] data = {2.1, 3.2};
    test.assignRow(1, new DenseVector(data));
    assertEquals("test[1][0]", 2.1, test.getQuick(1, 0));
    assertEquals("test[1][1]", 3.2, test.getQuick(1, 1));
  }

  public void testAssignRowCardinality() {
    double[] data = {2.1, 3.2, 4.3};
    try {
      test.assignRow(1, new DenseVector(data));
      fail("expecting cardinality exception");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testAssignColumn() {
    double[] data = {2.1, 3.2, 4.3};
    test.assignColumn(1, new DenseVector(data));
    assertEquals("test[0][1]", 2.1, test.getQuick(0, 1));
    assertEquals("test[1][1]", 3.2, test.getQuick(1, 1));
    assertEquals("test[2][1]", 4.3, test.getQuick(2, 1));
  }

  public void testAssignColumnCardinality() {
    double[] data = {2.1, 3.2};
    try {
      test.assignColumn(1, new DenseVector(data));
      fail("expecting cardinality exception");
    } catch (CardinalityException e) {
      assertTrue(true);
    }
  }

  public void testGetRow() {
    Vector row = test.getRow(1);
    assertEquals("row size", 2, row.getNumNondefaultElements());
  }

  public void testGetRowIndexUnder() {
    try {
      test.getRow(-1);
      fail("expecting index exception");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testGetRowIndexOver() {
    try {
      test.getRow(5);
      fail("expecting index exception");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testGetColumn() {
    Vector column = test.getColumn(1);
    assertEquals("row size", 3, column.getNumNondefaultElements());
  }

  public void testGetColumnIndexUnder() {
    try {
      test.getColumn(-1);
      fail("expecting index exception");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testGetColumnIndexOver() {
    try {
      test.getColumn(5);
      fail("expecting index exception");
    } catch (IndexException e) {
      assertTrue(true);
    }
  }

  public void testDetermitant() {
    Matrix m = matrixFactory(new double[][]{{1, 3, 4}, {5, 2, 3},
        {1, 4, 2}});
    assertEquals("determinant", 43.0, m.determinant());
  }

  public void testAsFormatString() {
    String string = test.asFormatString();
    int[] cardinality = {values.length, values[0].length};
    Matrix m = AbstractMatrix.decodeMatrix(string);
    for (int row = 0; row < cardinality[ROW]; row++) {
      for (int col = 0; col < cardinality[COL]; col++) {
        assertEquals("m[" + row + ',' + col + ']', test.get(row, col), m.get(
            row, col));
      }
    }
  }

  public void testLabelBindings() {
    Matrix m = matrixFactory(new double[][]{{1, 3, 4}, {5, 2, 3},
        {1, 4, 2}});
    assertNull("row bindings", m.getRowLabelBindings());
    assertNull("col bindings", m.getColumnLabelBindings());
    Map<String, Integer> rowBindings = new HashMap<String, Integer>();
    rowBindings.put("Fee", 0);
    rowBindings.put("Fie", 1);
    rowBindings.put("Foe", 2);
    m.setRowLabelBindings(rowBindings);
    assertEquals("row", rowBindings, m.getRowLabelBindings());
    Map<String, Integer> colBindings = new HashMap<String, Integer>();
    colBindings.put("Foo", 0);
    colBindings.put("Bar", 1);
    colBindings.put("Baz", 2);
    m.setColumnLabelBindings(colBindings);
    assertEquals("row", rowBindings, m.getRowLabelBindings());
    assertEquals("Fee", m.get(0, 1), m.get("Fee", "Bar"));

    double[] newrow = {9, 8, 7};
    m.set("Foe", newrow);
    assertEquals("FeeBaz", m.get(0, 2), m.get("Fee", "Baz"));
  }

  public void testSettingLabelBindings() {
    Matrix m = matrixFactory(new double[][]{{1, 3, 4}, {5, 2, 3},
        {1, 4, 2}});
    assertNull("row bindings", m.getRowLabelBindings());
    assertNull("col bindings", m.getColumnLabelBindings());
    m.set("Fee", "Foo", 1, 2, 9);
    assertNotNull("row", m.getRowLabelBindings());
    assertNotNull("row", m.getRowLabelBindings());
    assertEquals("Fee", 1, m.getRowLabelBindings().get("Fee").intValue());
    assertEquals("Fee", 2, m.getColumnLabelBindings().get("Foo").intValue());
    assertEquals("FeeFoo", m.get(1, 2), m.get("Fee", "Foo"));
    try {
      m.get("Fie", "Foe");
      fail("Expected UnboundLabelException");
    } catch (IndexException e) {
      fail("Expected UnboundLabelException");
    } catch (UnboundLabelException e) {
      assertTrue(true);
    }
  }

  public void testLabelBindingSerialization() {
    Matrix m = matrixFactory(new double[][]{{1, 3, 4}, {5, 2, 3},
        {1, 4, 2}});
    assertNull("row bindings", m.getRowLabelBindings());
    assertNull("col bindings", m.getColumnLabelBindings());
    Map<String, Integer> rowBindings = new HashMap<String, Integer>();
    rowBindings.put("Fee", 0);
    rowBindings.put("Fie", 1);
    rowBindings.put("Foe", 2);
    m.setRowLabelBindings(rowBindings);
    assertEquals("row", rowBindings, m.getRowLabelBindings());
    Map<String, Integer> colBindings = new HashMap<String, Integer>();
    colBindings.put("Foo", 0);
    colBindings.put("Bar", 1);
    colBindings.put("Baz", 2);
    m.setColumnLabelBindings(colBindings);
    String json = m.asFormatString();
    Matrix mm = AbstractMatrix.decodeMatrix(json);
    assertEquals("Fee", m.get(0, 1), mm.get("Fee", "Bar"));
  }

  public void testMatrixWritable() throws IOException {
    Matrix m = matrixFactory(new double[][]{{1, 3, 4}, {5, 2, 3},
        {1, 4, 2}});
    DataOutputBuffer out = new DataOutputBuffer();
    m.write(out);
    out.close();

    DataInputStream in = new DataInputStream(new ByteArrayInputStream(out
        .getData()));
    Matrix m2 = m.like();
    m2.readFields(in);
    in.close();
    assertEquals("row size", m.size()[ROW], m2.size()[ROW]);
    assertEquals("col size", m.size()[COL], m2.size()[COL]);
  }
}
