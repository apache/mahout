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

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

public class FileBasedSparseBinaryMatrixTest extends MahoutTestCase {

  /*
  // 10 million rows x 40 columns x 8 bytes = 3.2GB of data
  // we need >2GB to stress the file based matrix implementation
  private static final int ROWS = 10 * 1000 * 1000;
  private static final int COLUMNS = 1000;

  @Test
  public void testBigMatrix() throws IOException {
    // only run this test if -DrunSlowTests is used.  Also requires 4GB or more of heap.
    // assumeNotNull(System.getProperty("runSlowTests"));

    Matrix m0 = new SparseRowMatrix(ROWS, COLUMNS);
    Random gen = RandomUtils.getRandom();
    for (int i = 0; i < 1000; i++) {
      m0.set(gen.nextInt(ROWS), gen.nextInt(COLUMNS), matrixValue(i));
    }
    File f = File.createTempFile("foo", ".m");
    f.deleteOnExit();
    System.out.printf("Starting to write to %s\n", f.getAbsolutePath());
    FileBasedSparseBinaryMatrix.writeMatrix(f, m0);
    System.out.printf("done\n");
    System.out.printf("File is %.1f MB\n", f.length() / 1.0e6);

    FileBasedSparseBinaryMatrix m1 = new FileBasedSparseBinaryMatrix(ROWS, COLUMNS);
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
   */

  @Test
  public void testSetData() throws IOException {

    File f = File.createTempFile("matrix", ".m");
    f.deleteOnExit();

    Random gen = RandomUtils.getRandom();
    Matrix m0 = new SparseRowMatrix(10, 21);
    for (MatrixSlice row : m0) {
      int len = (int) Math.ceil(-15 * Math.log(1 - gen.nextDouble()));
      for (int i = 0; i < len; i++) {
        row.vector().set(gen.nextInt(21), 1);
      }
    }
    FileBasedSparseBinaryMatrix.writeMatrix(f, m0);

    FileBasedSparseBinaryMatrix m = new FileBasedSparseBinaryMatrix(10, 21);
    m.setData(f);

    for (MatrixSlice row : m) {
      Vector diff = row.vector().minus(m0.viewRow(row.index()));
      double error = diff.norm(1);
      if (error > 1.0e-14) {
        System.out.printf("%s\n", diff);
      }
      assertEquals(0, error, 1.0e-14);
    }
  }
}
