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

import com.google.common.collect.Maps;
import org.apache.mahout.math.function.Functions;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

public final class TestMatrixView extends MahoutTestCase {

  private static final int ROW = AbstractMatrix.ROW;
  private static final int COL = AbstractMatrix.COL;

  private final double[][] values = {{0.0, 1.1, 2.2}, {1.1, 2.2, 3.3},
    {3.3, 4.4, 5.5}, {5.5, 6.6, 7.7}, {7.7, 8.8, 9.9}};

  private Matrix test;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    int[] offset = {1, 1};
    int[] card = {3, 2};
    test = new MatrixView(new DenseMatrix(values), offset, card);
  }

  @Test
  public void testCardinality() {
    int[] c = test.size();
    assertEquals("row cardinality", values.length - 2, c[ROW]);
    assertEquals("col cardinality", values[0].length - 1, c[COL]);
  }

  @Test
  public void testCopy() {
    int[] c = test.size();
    Matrix copy = test.clone();
    assertTrue("wrong class", copy instanceof MatrixView);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            test.getQuick(row, col), copy.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testGetQuick() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1], test.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testLike() {
    Matrix like = test.like();
    assertTrue("type", like instanceof DenseMatrix);
    assertEquals("rows", test.size()[ROW], like.size()[ROW]);
    assertEquals("columns", test.size()[COL], like.size()[COL]);
  }

  @Test
  public void testLikeIntInt() {
    Matrix like = test.like(4, 4);
    assertTrue("type", like instanceof DenseMatrix);
    assertEquals("rows", 4, like.size()[ROW]);
    assertEquals("columns", 4, like.size()[COL]);
  }

  @Test
  public void testSetQuick() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        test.setQuick(row, col, 1.23);
        assertEquals("value[" + row + "][" + col + ']', 1.23, test.getQuick(
            row, col), EPSILON);
      }
    }
  }

  @Test
  public void testSize() {
    int[] c = test.getNumNondefaultElements();
    assertEquals("row size", values.length - 2, c[ROW]);
    assertEquals("col size", values[0].length - 1, c[COL]);
  }

  @Test
  public void testViewPart() {
    int[] offset = {1, 1};
    int[] size = {2, 1};
    Matrix view = test.viewPart(offset, size);
    int[] c = view.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 2][col + 2], view.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test(expected = IndexException.class)
  public void testViewPartCardinality() {
    int[] offset = {1, 1};
    int[] size = {3, 3};
    test.viewPart(offset, size);
  }

  @Test(expected = IndexException.class)
  public void testViewPartIndexOver() {
    int[] offset = {1, 1};
    int[] size = {2, 2};
    test.viewPart(offset, size);
  }

  @Test(expected = IndexException.class)
  public void testViewPartIndexUnder() {
    int[] offset = {-1, -1};
    int[] size = {2, 2};
    test.viewPart(offset, size);
  }

  @Test
  public void testAssignDouble() {
    int[] c = test.size();
    test.assign(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', 4.53, test.getQuick(
            row, col), EPSILON);
      }
    }
  }

  @Test
  public void testAssignDoubleArrayArray() {
    int[] c = test.size();
    test.assign(new double[3][2]);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', 0.0, test.getQuick(row,
            col), EPSILON);
      }
    }
  }

  @Test(expected = CardinalityException.class)
  public void testAssignDoubleArrayArrayCardinality() {
    int[] c = test.size();
    test.assign(new double[c[ROW] + 1][c[COL]]);
  }

  @Test
  public void testAssignMatrixBinaryFunction() {
    int[] c = test.size();
    test.assign(test, Functions.PLUS);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            2 * values[row + 1][col + 1], test.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test(expected = CardinalityException.class)
  public void testAssignMatrixBinaryFunctionCardinality() {
    test.assign(test.transpose(), Functions.PLUS);
  }

  @Test
  public void testAssignMatrix() {
    int[] c = test.size();
    Matrix value = test.like();
    value.assign(test);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            test.getQuick(row, col), value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test(expected = CardinalityException.class)
  public void testAssignMatrixCardinality() {
    test.assign(test.transpose());
  }

  @Test
  public void testAssignUnaryFunction() {
    int[] c = test.size();
    test.assign(Functions.NEGATE);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            -values[row + 1][col + 1], test.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testDivide() {
    int[] c = test.size();
    Matrix value = test.divide(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1] / 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testGet() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1], test.get(row, col), EPSILON);
      }
    }
  }

  @Test(expected = IndexException.class)
  public void testGetIndexUnder() {
    int[] c = test.size();
    for (int row = -1; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        test.get(row, col);
      }
    }
  }

  @Test(expected = IndexException.class)
  public void testGetIndexOver() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW] + 1; row++) {
      for (int col = 0; col < c[COL]; col++) {
        test.get(row, col);
      }
    }
  }

  @Test
  public void testMinus() {
    int[] c = test.size();
    Matrix value = test.minus(test);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', 0.0, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test(expected = CardinalityException.class)
  public void testMinusCardinality() {
    test.minus(test.transpose());
  }

  @Test
  public void testPlusDouble() {
    int[] c = test.size();
    Matrix value = test.plus(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1] + 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testPlusMatrix() {
    int[] c = test.size();
    Matrix value = test.plus(test);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1] * 2, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test(expected = CardinalityException.class)
  public void testPlusMatrixCardinality() {
    test.plus(test.transpose());
  }

  @Test(expected = IndexException.class)
  public void testSetUnder() {
    int[] c = test.size();
    for (int row = -1; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        test.set(row, col, 1.23);
      }
    }
  }

  @Test(expected = IndexException.class)
  public void testSetOver() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW] + 1; row++) {
      for (int col = 0; col < c[COL]; col++) {
        test.set(row, col, 1.23);
      }
    }
  }

  @Test
  public void testTimesDouble() {
    int[] c = test.size();
    Matrix value = test.times(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1] * 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testTimesMatrix() {
    int[] c = test.size();
    Matrix transpose = test.transpose();
    Matrix value = test.times(transpose);
    int[] v = value.size();
    assertEquals("rows", c[ROW], v[ROW]);
    assertEquals("cols", c[ROW], v[COL]);
    // TODO: check the math too, lazy
  }

  @Test(expected = CardinalityException.class)
  public void testTimesMatrixCardinality() {
    Matrix other = test.like(5, 8);
    test.times(other);
  }

  @Test
  public void testTranspose() {
    int[] c = test.size();
    Matrix transpose = test.transpose();
    int[] t = transpose.size();
    assertEquals("rows", c[COL], t[ROW]);
    assertEquals("cols", c[ROW], t[COL]);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            test.getQuick(row, col), transpose.getQuick(col, row), EPSILON);
      }
    }
  }

  @Test
  public void testZSum() {
    double sum = test.zSum();
    assertEquals("zsum", 29.7, sum, EPSILON);
  }

  @Test
  public void testAssignRow() {
    double[] data = {2.1, 3.2};
    test.assignRow(1, new DenseVector(data));
    assertEquals("test[1][0]", 2.1, test.getQuick(1, 0), EPSILON);
    assertEquals("test[1][1]", 3.2, test.getQuick(1, 1), EPSILON);
  }

  @Test(expected = CardinalityException.class)
  public void testAssignRowCardinality() {
    double[] data = {2.1, 3.2, 4.3};
    test.assignRow(1, new DenseVector(data));
  }

  @Test
  public void testAssignColumn() {
    double[] data = {2.1, 3.2, 4.3};
    test.assignColumn(1, new DenseVector(data));
    assertEquals("test[0][1]", 2.1, test.getQuick(0, 1), EPSILON);
    assertEquals("test[1][1]", 3.2, test.getQuick(1, 1), EPSILON);
    assertEquals("test[2][1]", 4.3, test.getQuick(2, 1), EPSILON);
  }

  @Test(expected = CardinalityException.class)
  public void testAssignColumnCardinality() {
    double[] data = {2.1, 3.2};
    test.assignColumn(1, new DenseVector(data));
  }

  @Test
  public void testGetRow() {
    Vector row = test.getRow(1);
    assertEquals("row size", 2, row.getNumNondefaultElements());
  }

  @Test(expected = IndexException.class)
  public void testGetRowIndexUnder() {
    test.getRow(-1);
  }

  @Test(expected = IndexException.class)
  public void testGetRowIndexOver() {
    test.getRow(5);
  }

  @Test
  public void testGetColumn() {
    Vector column = test.getColumn(1);
    assertEquals("row size", 3, column.getNumNondefaultElements());
    int i = 0;
    for (double x : new double[]{3.3, 5.5, 7.7}) {
      assertEquals(x, column.get(i++), 0);
    }
  }

  @Test(expected = IndexException.class)
  public void testGetColumnIndexUnder() {
    test.getColumn(-1);
  }

  @Test(expected = IndexException.class)
  public void testGetColumnIndexOver() {
    test.getColumn(5);
  }

  @Test
  public void testLabelBindings() {
    assertNull("row bindings", test.getRowLabelBindings());
    assertNull("col bindings", test.getColumnLabelBindings());
    Map<String, Integer> rowBindings = Maps.newHashMap();
    rowBindings.put("Fee", 0);
    rowBindings.put("Fie", 1);
    test.setRowLabelBindings(rowBindings);
    assertEquals("row", rowBindings, test.getRowLabelBindings());
    Map<String, Integer> colBindings = Maps.newHashMap();
    colBindings.put("Foo", 0);
    colBindings.put("Bar", 1);
    test.setColumnLabelBindings(colBindings);
    assertEquals("row", rowBindings, test.getRowLabelBindings());
    assertEquals("Fee", test.get(0, 1), test.get("Fee", "Bar"), EPSILON);

    double[] newrow = {9, 8};
    test.set("Fie", newrow);
    assertEquals("FeeBar", test.get(0, 1), test.get("Fee", "Bar"), EPSILON);
  }

  @Test(expected = IllegalStateException.class)
  public void testSettingLabelBindings() {
    assertNull("row bindings", test.getRowLabelBindings());
    assertNull("col bindings", test.getColumnLabelBindings());
    test.set("Fee", "Foo", 1, 1, 9);
    assertNotNull("row", test.getRowLabelBindings());
    assertNotNull("row", test.getRowLabelBindings());
    assertEquals("Fee", 1, test.getRowLabelBindings().get("Fee").intValue());
    assertEquals("Foo", 1, test.getColumnLabelBindings().get("Foo").intValue());
    assertEquals("FeeFoo", test.get(1, 1), test.get("Fee", "Foo"), EPSILON);
    test.get("Fie", "Foe");
  }

  @Test
  public void testLabelBindingSerialization() {
    assertNull("row bindings", test.getRowLabelBindings());
    assertNull("col bindings", test.getColumnLabelBindings());
    Map<String, Integer> rowBindings = Maps.newHashMap();
    rowBindings.put("Fee", 0);
    rowBindings.put("Fie", 1);
    rowBindings.put("Foe", 2);
    test.setRowLabelBindings(rowBindings);
    assertEquals("row", rowBindings, test.getRowLabelBindings());
    Map<String, Integer> colBindings = Maps.newHashMap();
    colBindings.put("Foo", 0);
    colBindings.put("Bar", 1);
    colBindings.put("Baz", 2);
    test.setColumnLabelBindings(colBindings);
    assertEquals("col", colBindings, test.getColumnLabelBindings());
  }

}
