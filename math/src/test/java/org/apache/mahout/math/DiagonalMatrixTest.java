/*
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
import org.junit.Assert;
import org.junit.Test;

import java.util.Iterator;

public class DiagonalMatrixTest extends MahoutTestCase {
  @Test
  public void testBasics() {
    DiagonalMatrix a = new DiagonalMatrix(new double[]{1, 2, 3, 4});

    assertEquals(0, a.viewDiagonal().minus(new DenseVector(new double[]{1, 2, 3, 4})).norm(1), 1.0e-10);
    assertEquals(0, a.viewPart(0, 3, 0, 3).viewDiagonal().minus(
      new DenseVector(new double[]{1, 2, 3})).norm(1), 1.0e-10);

    assertEquals(4, a.get(3, 3), 1.0e-10);

    Matrix m = new DenseMatrix(4, 4);
    m.assign(a);

    assertEquals(0, m.minus(a).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);

    assertEquals(0, m.transpose().times(m).minus(a.transpose().times(a)).aggregate(
      Functions.PLUS, Functions.ABS), 1.0e-10);
    assertEquals(0, m.plus(m).minus(a.plus(a)).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);

    m = new DenseMatrix(new double[][]{{1, 2, 3, 4}, {5, 6, 7, 8}});

    assertEquals(100, a.timesLeft(m).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);
    assertEquals(100, a.times(m.transpose()).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);
  }

  @Test
  public void testSparsity() {
    Vector d = new DenseVector(10);
    for (int i = 0; i < 10; i++) {
      d.set(i, i * i);
    }
    DiagonalMatrix m = new DiagonalMatrix(d);

    Assert.assertFalse(m.viewRow(0).isDense());
    Assert.assertFalse(m.viewColumn(0).isDense());

    for (int i = 0; i < 10; i++) {
      assertEquals(i * i, m.viewRow(i).zSum(), 0);
      assertEquals(i * i, m.viewRow(i).get(i), 0);

      assertEquals(i * i, m.viewColumn(i).zSum(), 0);
      assertEquals(i * i, m.viewColumn(i).get(i), 0);
    }

    Iterator<Vector.Element> ix = m.viewRow(7).nonZeroes().iterator();
    assertTrue(ix.hasNext());
    Vector.Element r = ix.next();
    assertEquals(7, r.index());
    assertEquals(49, r.get(), 0);
    assertFalse(ix.hasNext());

    assertEquals(0, m.viewRow(5).get(3), 0);
    assertEquals(0, m.viewColumn(8).get(3), 0);

    m.viewRow(3).set(3, 1);
    assertEquals(1, m.get(3, 3), 0);

    for (Vector.Element element : m.viewRow(6).all()) {
      if (element.index() == 6) {
        assertEquals(36, element.get(), 0);
      }                                    else {
        assertEquals(0, element.get(), 0);
      }
    }
  }
}
