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

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.random.MultiNormal;
import org.junit.Assume;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class FileBasedMatrixTest extends MahoutTestCase {
  // 10 million rows x 40 columns x 8 bytes = 3.2GB of data
  // we need >2GB to stress the file based matrix implementation
  private static final int ROWS = 10 * 1000 * 1000;
  private static final int COLUMNS = 40;

  @Test
  public void testBigMatrix() throws IOException {
    // only run this test if -DrunSlowTests is used.  Also requires 4GB or more of heap.
    Assume.assumeNotNull(System.getProperty("runSlowTests"));

    Matrix m0 = new SparseRowMatrix(ROWS, COLUMNS);
    Random gen = RandomUtils.getRandom();
    for (int i = 0; i < 1000; i++) {
      m0.set(gen.nextInt(ROWS), gen.nextInt(COLUMNS), matrixValue(i));
    }
    File f = File.createTempFile("foo", ".m");
    f.deleteOnExit();
    System.out.printf("Starting to write to %s\n", f.getAbsolutePath());
    FileBasedMatrix.writeMatrix(f, m0);
    System.out.printf("done\n");
    System.out.printf("File is %.1f MB\n", f.length() / 1.0e6);

    FileBasedMatrix m1 = new FileBasedMatrix(ROWS, COLUMNS);
    System.out.printf("Starting read\n");
    m1.setData(f, false);
    gen = RandomUtils.getRandom();
    for (int i = 0; i < 1000; i++) {
      assertEquals(matrixValue(i), m1.get(gen.nextInt(ROWS), gen.nextInt(COLUMNS)), 0.0);
    }
    System.out.printf("done\n");
  }

  private static int matrixValue(int i) {
    return (i * 88513) % 10000;
  }

  @Test
  public void testSetData() throws IOException {
    File f = File.createTempFile("matrix", ".m");
    f.deleteOnExit();

    Matrix m0 = new DenseMatrix(100000, 30);
    MultiNormal gen = new MultiNormal(30);
    for (MatrixSlice row : m0) {
      row.vector().assign(gen.sample());
    }
    FileBasedMatrix.writeMatrix(f, m0);

    FileBasedMatrix m = new FileBasedMatrix(100000, 30);
    m.setData(f, true);

    assertEquals(0, m0.minus(m).aggregate(Functions.MAX, Functions.ABS), 1.0e-8);

    int i = 0;
    for (MatrixSlice row : m) {
      assertEquals(0, row.vector().minus(m0.viewRow(i++)).norm(1), 1.0e-8);
    }
  }
}
