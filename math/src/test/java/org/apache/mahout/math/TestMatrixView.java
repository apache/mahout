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
    assertEquals("row cardinality", values.length - 2, test.rowSize());
    assertEquals("col cardinality", values[0].length - 1, test.columnSize());
  }

  @Test
  public void testCopy() {
    Matrix copy = test.clone();
    assertTrue("wrong class", copy instanceof MatrixView);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            test.getQuick(row, col), copy.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testGetQuick() {
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1], test.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testLike() {
    Matrix like = test.like();
    assertTrue("type", like instanceof DenseMatrix);
    assertEquals("rows", test.rowSize(), like.rowSize());
    assertEquals("columns", test.columnSize(), like.columnSize());
  }

  @Test
  public void testLikeIntInt() {
    Matrix like = test.like(4, 4);
    assertTrue("type", like instanceof DenseMatrix);
    assertEquals("rows", 4, like.rowSize());
    assertEquals("columns", 4, like.columnSize());
  }

  @Test
  public void testSetQuick() {
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        test.setQuick(row, col, 1.23);
        assertEquals("value[" + row + "][" + col + ']', 1.23, test.getQuick(
            row, col), EPSILON);
      }
    }
  }

  @Test
  public void testSize() {
    assertEquals("row size", values.length - 2, test.rowSize());
    assertEquals("col size", values[0].length - 1, test.columnSize());
  }

  @Test
  public void testViewPart() {
    int[] offset = {1, 1};
    int[] size = {2, 1};
    Matrix view = test.viewPart(offset, size);
    for (int row = 0; row < view.rowSize(); row++) {
      for (int col = 0; col < view.columnSize(); col++) {
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
    test.assign(4.53);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']', 4.53, test.getQuick(
            row, col), EPSILON);
      }
    }
  }

  @Test
  public void testAssignDoubleArrayArray() {
    test.assign(new double[3][2]);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']', 0.0, test.getQuick(row,
            col), EPSILON);
      }
    }
  }

  @Test(expected = CardinalityException.class)
  public void testAssignDoubleArrayArrayCardinality() {
    test.assign(new double[test.rowSize() + 1][test.columnSize()]);
  }

  @Test
  public void testAssignMatrixBinaryFunction() {
    test.assign(test, Functions.PLUS);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
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
    Matrix value = test.like();
    value.assign(test);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
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
    test.assign(Functions.NEGATE);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            -values[row + 1][col + 1], test.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testDivide() {
    Matrix value = test.divide(4.53);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1] / 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testGet() {
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1], test.get(row, col), EPSILON);
      }
    }
  }

  @Test(expected = IndexException.class)
  public void testGetIndexUnder() {
    for (int row = -1; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        test.get(row, col);
      }
    }
  }

  @Test(expected = IndexException.class)
  public void testGetIndexOver() {
    for (int row = 0; row < test.rowSize() + 1; row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        test.get(row, col);
      }
    }
  }

  @Test
  public void testMinus() {
    Matrix value = test.minus(test);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
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
    Matrix value = test.plus(4.53);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1] + 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testPlusMatrix() {
    Matrix value = test.plus(test);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
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
    for (int row = -1; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        test.set(row, col, 1.23);
      }
    }
  }

  @Test(expected = IndexException.class)
  public void testSetOver() {
    for (int row = 0; row < test.rowSize() + 1; row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        test.set(row, col, 1.23);
      }
    }
  }

  @Test
  public void testTimesDouble() {
    Matrix value = test.times(4.53);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1] * 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testTimesMatrix() {
    Matrix transpose = test.transpose();
    Matrix value = test.times(transpose);
    assertEquals("rows", test.rowSize(), value.rowSize());
    assertEquals("cols", test.rowSize(), value.columnSize());
    // TODO: check the math too, lazy
  }

  @Test(expected = CardinalityException.class)
  public void testTimesMatrixCardinality() {
    Matrix other = test.like(5, 8);
    test.times(other);
  }

  @Test
  public void testTranspose() {
    Matrix transpose = test.transpose();
    assertEquals("rows", test.columnSize(), transpose.rowSize());
    assertEquals("cols", test.rowSize(), transpose.columnSize());
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
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
  public void testViewRow() {
    Vector row = test.viewRow(1);
    assertEquals("row size", 2, row.getNumNondefaultElements());
  }

  @Test(expected = IndexException.class)
  public void testViewRowIndexUnder() {
    test.viewRow(-1);
  }

  @Test(expected = IndexException.class)
  public void testViewRowIndexOver() {
    test.viewRow(5);
  }

  @Test
  public void testViewColumn() {
    Vector column = test.viewColumn(1);
    assertEquals("row size", 3, column.getNumNondefaultElements());
    int i = 0;
    for (double x : new double[]{3.3, 5.5, 7.7}) {
      assertEquals(x, column.get(i++), 0);
    }
  }

  @Test(expected = IndexException.class)
  public void testViewColumnIndexUnder() {
    test.viewColumn(-1);
  }

  @Test(expected = IndexException.class)
  public void testViewColumnIndexOver() {
    test.viewColumn(5);
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
