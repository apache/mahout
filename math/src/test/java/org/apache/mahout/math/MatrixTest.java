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
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;
import org.junit.Before;
import org.junit.Test;

import java.util.Iterator;
import java.util.Map;
import java.util.Random;

public abstract class MatrixTest extends MahoutTestCase {

  protected static final int ROW = AbstractMatrix.ROW;

  protected static final int COL = AbstractMatrix.COL;

  private final double[][] values = {{1.1, 2.2}, {3.3, 4.4},
      {5.5, 6.6}};

  private final double[] vectorAValues = {1.0 / 1.1, 2.0 / 1.1};

  //protected final double[] vectorBValues = {5.0, 10.0, 100.0};

  protected Matrix test;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    test = matrixFactory(values);
  }

  public abstract Matrix matrixFactory(double[][] values);

  @Test
  public void testCardinality() {
    int[] c = test.size();
    assertEquals("row cardinality", values.length, c[ROW]);
    assertEquals("col cardinality", values[0].length, c[COL]);
  }

  @Test
  public void testCopy() {
    int[] c = test.size();
    Matrix copy = test.clone();
    assertSame("wrong class", copy.getClass(), test.getClass());
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            test.getQuick(row, col), copy.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testIterate() {
    Iterator<MatrixSlice> it = test.iterator();
    MatrixSlice m;
    while(it.hasNext() && (m = it.next()) != null) {
      Vector v = m.vector();
      Vector w = test instanceof SparseColumnMatrix ? test.getColumn(m.index()) : test.getRow(m.index());
      assertEquals("iterator: " + v + ", randomAccess: " + w, v, w);
    }
  }

  @Test
  public void testGetQuick() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', values[row][col], test
            .getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testLike() {
    Matrix like = test.like();
    assertSame("type", like.getClass(), test.getClass());
    assertEquals("rows", test.size()[ROW], like.size()[ROW]);
    assertEquals("columns", test.size()[COL], like.size()[COL]);
  }

  @Test
  public void testLikeIntInt() {
    Matrix like = test.like(4, 4);
    assertSame("type", like.getClass(), test.getClass());
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
    int[] c = view.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row + 1][col + 1], view.getQuick(row, col), EPSILON);
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
    test.assign(Functions.mult(-1));
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', -values[row][col], test
            .getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testRowView() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW]; row++) {
      assertEquals(0.0, test.getRow(row).minus(test.viewRow(row)).norm(1), 0);
    }

    assertEquals(c[COL], test.viewRow(3).size());
    assertEquals(c[COL], test.viewRow(5).size());

    Random gen = RandomUtils.getRandom();
    for (int row = 0; row < c[ROW]; row++) {
      int j = gen.nextInt(c[COL]);
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
    int[] c = test.size();
    for (int col = 0; col < c[COL]; col++) {
      assertEquals(0.0, test.getColumn(col).minus(test.viewColumn(col)).norm(1), 0);
    }

    assertEquals(c[ROW], test.viewColumn(3).size());
    assertEquals(c[ROW], test.viewColumn(5).size());

    Random gen = RandomUtils.getRandom();
    for (int col = 0; col < c[COL]; col++) {
      int j = gen.nextInt(c[COL]);
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
      assertEquals(test.getRow(i).zSum(), v.get(i), EPSILON);
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
      assertEquals(test.getColumn(i).zSum(), v.get(i), EPSILON);
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
    int[] c = test.size();
    Matrix value = test.divide(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row][col] / 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testGet() {
    int[] c = test.size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']', values[row][col], test
            .get(row, col), EPSILON);
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
    int[] c = test.size();
    Matrix value = test.plus(4.53);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        assertEquals("value[" + row + "][" + col + ']',
            values[row][col] + 4.53, value.getQuick(row, col), EPSILON);
      }
    }
  }

  @Test
  public void testPlusMatrix() {
    int[] c = test.size();
    Matrix value = test.plus(test);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
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
            values[row][col] * 4.53, value.getQuick(row, col), EPSILON);
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
  public void testDetermitant() {
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

  @Test(expected = UnboundLabelException.class)
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
