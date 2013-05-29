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
import java.util.Map;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;
import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.Maps;

public abstract class MatrixTest extends MahoutTestCase {

  protected static final int ROW = AbstractMatrix.ROW;

  protected static final int COL = AbstractMatrix.COL;

  private final double[][] values = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};

  private final double[] vectorAValues = {1.0 / 1.1, 2.0 / 1.1};
  private Matrix test;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    test = matrixFactory(values);
  }

  public abstract Matrix matrixFactory(double[][] values);

  @Test
  public void testCardinality() {
    assertEquals("row cardinality", values.length, test.rowSize());
    assertEquals("col cardinality", values[0].length, test.columnSize());
  }

  @Test
  public void testCopy() {
    Matrix copy = test.clone();
    assertSame("wrong class", copy.getClass(), test.getClass());
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            test.getQuick(row, col), copy.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testClone() {
    double oldValue = 1.23;
    double newValue = 2.34;
    double[][] values = {{oldValue, 3}, {3, 5}, {7, 9}};
    Matrix matrix = matrixFactory(values);
    Matrix clone = matrix.clone();
    clone.set(0, 0, newValue);
    //test whether the update in the clone is reflected in the original matrix
    assertEquals("Matrix clone is not independent of the original",
      oldValue, matrix.get(0, 0), EPSILON);
  }

  @Test
  public void testIterate() {
    Iterator<MatrixSlice> it = test.iterator();
    MatrixSlice m;
    while (it.hasNext() && (m = it.next()) != null) {
      Vector v = m.vector();
      Vector w = test instanceof SparseColumnMatrix ? test.viewColumn(m.index()) : test.viewRow(m.index());
      assertEquals("iterator: " + v + ", randomAccess: " + w, v, w);
    }
  }

  @Test
  public void testGetQuick() {
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']', values[row][col], test
            .getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testLike() {
    Matrix like = test.like();
    assertSame("type", like.getClass(), test.getClass());
    assertEquals("rows", test.rowSize(), like.rowSize());
    assertEquals("columns", test.columnSize(), like.columnSize());
  }

  @Test
  public void testLikeIntInt() {
    Matrix like = test.like(4, 4);
    assertSame("type", like.getClass(), test.getClass());
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
    int[] c = test.getNumNondefaultElements();
    assertEquals("row size", values.length, c[ROW]);
    assertEquals("col size", values[0].length, c[COL]);
  }

  @Test
  public void testViewPart() {
    int[] offset = {1, 1};
    int[] size = {2, 1};
    Matrix view = test.viewPart(offset, size);
    assertEquals(2, view.rowSize());
    assertEquals(1, view.columnSize());
    for (int row = 0; row < view.rowSize(); row++) {
      for (int col = 0; col < view.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1], view.get(row, col), EPSILON);
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

  /** Tests MAHOUT-1046 */
  @Test
  public void testMatrixViewBug() {
    Matrix m = test.viewPart(0, 3, 0, 2);
    // old bug would blow cookies with an index exception here.
    m = m.viewPart(2, 1, 0, 1);
    assertEquals(5.5, m.zSum(), 0);
  }

  @Test
  public void testAssignMatrixBinaryFunction() {
    test.assign(test, Functions.PLUS);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']', 2 * values[row][col],
            test.getQuick(row, col), EPSILON);
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
    test.assign(Functions.mult(-1));
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']', -values[row][col], test
            .getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testRowView() {
    assertEquals(test.columnSize(), test.viewRow(1).size());
    assertEquals(test.columnSize(), test.viewRow(2).size());

    Random gen = RandomUtils.getRandom();
    for (int row = 0; row < test.rowSize(); row++) {
      int j = gen.nextInt(test.columnSize());
      double old = test.get(row, j);
      double v = gen.nextGaussian();
      test.viewRow(row).set(j, v);
      assertEquals(v, test.get(row, j), 0);
      assertEquals(v, test.viewRow(row).get(j), 0);
      test.set(row, j, old);
      assertEquals(old, test.get(row, j), 0);
      assertEquals(old, test.viewRow(row).get(j), 0);
    }
  }

  @Test
  public void testColumnView() {
    assertEquals(test.rowSize(), test.viewColumn(0).size());
    assertEquals(test.rowSize(), test.viewColumn(1).size());

    Random gen = RandomUtils.getRandom();
    for (int col = 0; col < test.columnSize(); col++) {
      int j = gen.nextInt(test.columnSize());
      double old = test.get(col, j);
      double v = gen.nextGaussian();
      test.viewColumn(col).set(j, v);
      assertEquals(v, test.get(j, col), 0);
      assertEquals(v, test.viewColumn(col).get(j), 0);
      test.set(j, col, old);
      assertEquals(old, test.get(j, col), 0);
      assertEquals(old, test.viewColumn(col).get(j), 0);
    }
  }

  @Test
  public void testAggregateRows() {
    Vector v = test.aggregateRows(new VectorFunction() {
      @Override
      public double apply(Vector v) {
        return v.zSum();
      }
    });

    for (int i = 0; i < test.numRows(); i++) {
      assertEquals(test.viewRow(i).zSum(), v.get(i), EPSILON);
    }
  }

  @Test
  public void testAggregateCols() {
    Vector v = test.aggregateColumns(new VectorFunction() {
      @Override
      public double apply(Vector v) {
        return v.zSum();
      }
    });

    for (int i = 0; i < test.numCols(); i++) {
      assertEquals(test.viewColumn(i).zSum(), v.get(i), EPSILON);
    }
  }

  @Test
  public void testAggregate() {
    double total = test.aggregate(Functions.PLUS, Functions.IDENTITY);
    assertEquals(test.aggregateRows(new VectorFunction() {
      @Override
      public double apply(Vector v) {
        return v.zSum();
      }
    }).zSum(), total, EPSILON);
  }

  @Test
  public void testDivide() {
    Matrix value = test.divide(4.53);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row][col] / 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testGet() {
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']', values[row][col], test
            .get(row, col), EPSILON);
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
        assertEquals("value[" + row + "][" + col + ']', 0.0, value.getQuick(
            row, col), EPSILON);
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
            values[row][col] + 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testPlusMatrix() {
    Matrix value = test.plus(test);
    for (int row = 0; row < test.rowSize(); row++) {
      for (int col = 0; col < test.columnSize(); col++) {
        assertEquals("value[" + row + "][" + col + ']', values[row][col] * 2,
            value.getQuick(row, col), EPSILON);
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
            values[row][col] * 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testTimesMatrix() {
    Matrix transpose = test.transpose();
    Matrix value = test.times(transpose);
    assertEquals("rows", test.rowSize(), value.rowSize());
    assertEquals("cols", test.rowSize(), value.columnSize());

    Matrix expected = new DenseMatrix(new double[][]{{5.0, 11.0, 17.0},
        {11.0, 25.0, 39.0}, {17.0, 39.0, 61.0}}).times(1.21);

    for (int i = 0; i < expected.numCols(); i++) {
      for (int j = 0; j < expected.numRows(); j++) {
        assertTrue("Matrix times transpose not correct: " + i + ", " + j
            + "\nexpected:\n\t" + expected + "\nactual:\n\t"
            + value,
            Math.abs(expected.get(i, j) - value.get(i, j)) < 1.0e-12);
      }
    }

    Matrix timestest = new DenseMatrix(10, 1);
    /* will throw ArrayIndexOutOfBoundsException exception without MAHOUT-26 */
    timestest.transpose().times(timestest);
  }

  @Test(expected = CardinalityException.class)
  public void testTimesVector() {
    Vector vectorA = new DenseVector(vectorAValues);
    Vector testTimesVectorA = test.times(vectorA);
    Vector expected = new DenseVector(new double[]{5.0, 11.0, 17.0});
    assertTrue("Matrix times vector not equals: " + vectorA + " != " + testTimesVectorA,
        expected.minus(testTimesVectorA).norm(2) < 1.0e-12);
    test.times(testTimesVectorA);
  }

  @Test
  public void testTimesSquaredTimesVector() {
    Vector vectorA = new DenseVector(vectorAValues);
    Vector ttA = test.timesSquared(vectorA);
    Vector ttASlow = test.transpose().times(test.times(vectorA));
    assertTrue("M'Mv != M.timesSquared(v): " + ttA + " != " + ttASlow,
        ttASlow.minus(ttA).norm(2) < 1.0e-12);

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
    assertEquals("zsum", 23.1, sum, EPSILON);
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

    //create a matrix with an unassigned row 0
    Matrix matrix = new SparseMatrix(1, 1);
    Vector view = matrix.viewRow(0);
    final double value = 1.23;
    view.assign(value);
    //test whether the update in the view is reflected in the matrix
    assertEquals("Matrix value", view.getQuick(0), matrix.getQuick(0, 0), EPSILON);
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
  public void testDeterminant() {
    Matrix m = matrixFactory(new double[][]{{1, 3, 4}, {5, 2, 3},
        {1, 4, 2}});
    assertEquals("determinant", 43.0, m.determinant(), EPSILON);
  }

  @Test
  public void testLabelBindings() {
    Matrix m = matrixFactory(new double[][]{{1, 3, 4}, {5, 2, 3},
        {1, 4, 2}});
    assertNull("row bindings", m.getRowLabelBindings());
    assertNull("col bindings", m.getColumnLabelBindings());
    Map<String, Integer> rowBindings = Maps.newHashMap();
    rowBindings.put("Fee", 0);
    rowBindings.put("Fie", 1);
    rowBindings.put("Foe", 2);
    m.setRowLabelBindings(rowBindings);
    assertEquals("row", rowBindings, m.getRowLabelBindings());
    Map<String, Integer> colBindings = Maps.newHashMap();
    colBindings.put("Foo", 0);
    colBindings.put("Bar", 1);
    colBindings.put("Baz", 2);
    m.setColumnLabelBindings(colBindings);
    assertEquals("row", rowBindings, m.getRowLabelBindings());
    assertEquals("Fee", m.get(0, 1), m.get("Fee", "Bar"), EPSILON);

    double[] newrow = {9, 8, 7};
    m.set("Foe", newrow);
    assertEquals("FeeBaz", m.get(0, 2), m.get("Fee", "Baz"), EPSILON);
  }

  @Test(expected = IllegalStateException.class)
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
    assertEquals("FeeFoo", m.get(1, 2), m.get("Fee", "Foo"), EPSILON);
    m.get("Fie", "Foe");
  }

  @Test
  public void testLabelBindingSerialization() {
    Matrix m = matrixFactory(new double[][]{{1, 3, 4}, {5, 2, 3},
        {1, 4, 2}});
    assertNull("row bindings", m.getRowLabelBindings());
    assertNull("col bindings", m.getColumnLabelBindings());
    Map<String, Integer> rowBindings = Maps.newHashMap();
    rowBindings.put("Fee", 0);
    rowBindings.put("Fie", 1);
    rowBindings.put("Foe", 2);
    m.setRowLabelBindings(rowBindings);
    assertEquals("row", rowBindings, m.getRowLabelBindings());
    Map<String, Integer> colBindings = Maps.newHashMap();
    colBindings.put("Foo", 0);
    colBindings.put("Bar", 1);
    colBindings.put("Baz", 2);
    m.setColumnLabelBindings(colBindings);
    assertEquals("col", colBindings, m.getColumnLabelBindings());
  }
}
