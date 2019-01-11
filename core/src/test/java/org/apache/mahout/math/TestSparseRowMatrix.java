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

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.jet.random.Gamma;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

public final class TestSparseRowMatrix extends MatrixTest {

  @Override
  public Matrix matrixFactory(double[][] values) {
    Matrix matrix = new SparseRowMatrix(values.length, values[0].length);
    for (int row = 0; row < matrix.rowSize(); row++) {
      for (int col = 0; col < matrix.columnSize(); col++) {
        matrix.setQuick(row, col, values[row][col]);
      }
    }
    return matrix;
  }


  @Test(timeout=50000)
  public void testTimesSparseEfficiency() {
    Random raw = RandomUtils.getRandom();
    Gamma gen = new Gamma(0.1, 0.1, raw);

    // build two large sequential sparse matrices and multiply them
    Matrix x = new SparseRowMatrix(1000, 2000, false);
    for (int i = 0; i < 1000; i++) {
      int[] values = new int[1000];
      for (int k = 0; k < 1000; k++) {
        int j = (int) Math.min(1000, gen.nextDouble());
        values[j]++;
      }
      for (int j = 0; j < 1000; j++) {
        if (values[j] > 0) {
          x.set(i, j, values[j]);
        }
      }
    }

    Matrix y = new SparseRowMatrix(2000, 1000, false);
    for (int i = 0; i < 2000; i++) {
      int[] values = new int[1000];
      for (int k = 0; k < 1000; k++) {
        int j = (int) Math.min(1000, gen.nextDouble());
        values[j]++;
      }
      for (int j = 0; j < 1000; j++) {
        if (values[j] > 0) {
          y.set(i, j, values[j]);
        }
      }
    }

    long t0 = System.nanoTime();
    Matrix z = x.times(y);
    double elapsedTime = (System.nanoTime() - t0) * 1e-6;
    System.out.printf("done in %.1f ms\n", elapsedTime);

    for (int k = 0; k < 1000; k++) {
      int i = (int) (-10 * Math.log(raw.nextDouble()));
      int j = (int) (-10 * Math.log(raw.nextDouble()));
      Assert.assertEquals(x.viewRow(i).dot(y.viewColumn(j)), z.get(i, j), 1e-12);
    }
  }

  @Test(timeout=50000)
  public void testTimesDenseEfficiency() {
    Random raw = RandomUtils.getRandom();
    Gamma gen = new Gamma(0.1, 0.1, raw);

    // build a sequential sparse matrix and a dense matrix and multiply them
    Matrix x = new SparseRowMatrix(1000, 2000, false);
    for (int i = 0; i < 1000; i++) {
      int[] values = new int[1000];
      for (int k = 0; k < 1000; k++) {
        int j = (int) Math.min(1000, gen.nextDouble());
        values[j]++;
      }
      for (int j = 0; j < 1000; j++) {
        if (values[j] > 0) {
          x.set(i, j, values[j]);
        }
      }
    }

    Matrix y = new DenseMatrix(2000, 20);
    for (int i = 0; i < 2000; i++) {
      for (int j = 0; j < 20; j++) {
        y.set(i, j, raw.nextDouble());
      }
    }

    long t0 = System.nanoTime();
    Matrix z = x.times(y);
    double elapsedTime = (System.nanoTime() - t0) * 1e-6;
    System.out.printf("done in %.1f ms\n", elapsedTime);

    for (int i = 0; i < 1000; i++) {
      for (int j = 0; j < 20; j++) {
        Assert.assertEquals(x.viewRow(i).dot(y.viewColumn(j)), z.get(i, j), 1e-12);
      }
    }
  }

  @Test(timeout=50000)
  public void testTimesOtherSparseEfficiency() {
    Random raw = RandomUtils.getRandom();
    Gamma gen = new Gamma(0.1, 0.1, raw);

    // build a sequential sparse matrix and a diagonal matrix and multiply them
    Matrix x = new SparseRowMatrix(1000, 2000, false);
    for (int i = 0; i < 1000; i++) {
      int[] values = new int[1000];
      for (int k = 0; k < 1000; k++) {
        int j = (int) Math.min(1000, gen.nextDouble());
        values[j]++;
      }
      for (int j = 0; j < 1000; j++) {
        if (values[j] > 0) {
          x.set(i, j, values[j]);
        }
      }
    }

    Vector d = new DenseVector(2000).assign(Functions.random());
    Matrix y = new DiagonalMatrix(d);

    long t0 = System.nanoTime();
    Matrix z = x.times(y);
    double elapsedTime = (System.nanoTime() - t0) * 1e-6;
    System.out.printf("done in %.1f ms\n", elapsedTime);

    for (MatrixSlice row : z) {
      for (Vector.Element element : row.nonZeroes()) {
        assertEquals(x.get(row.index(), element.index()) * d.get(element.index()), element.get(), 1e-12);
      }
    }
  }


  @Test(timeout=50000)
  public void testTimesCorrect() {
    Random raw = RandomUtils.getRandom();

    // build two large sequential sparse matrices and multiply them
    Matrix x = new SparseRowMatrix(100, 2000, false)
      .assign(Functions.random());

    Matrix y = new SparseRowMatrix(2000, 100, false)
      .assign(Functions.random());

    Matrix xd = new DenseMatrix(100, 2000).assign(x);
    Matrix yd = new DenseMatrix(2000, 100).assign(y);
    assertEquals(0, xd.times(yd).minus(x.times(y)).aggregate(Functions.PLUS, Functions.ABS), 1e-15);
    assertEquals(0, x.times(yd).minus(x.times(y)).aggregate(Functions.PLUS, Functions.ABS), 1e-15);
    assertEquals(0, xd.times(y).minus(x.times(y)).aggregate(Functions.PLUS, Functions.ABS), 1e-15);
  }
}
