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

public class TestSparseColumnMatrix extends MatrixTest {

  public TestSparseColumnMatrix(String name) {
    super(name);
  }

  @Override
  public Matrix matrixFactory(double[][] values) {
    int[] cardinality = {values.length, values[0].length};
    Matrix matrix = new SparseColumnMatrix(cardinality);
    for (int row = 0; row < cardinality[ROW]; row++) {
      for (int col = 0; col < cardinality[COL]; col++) {
        matrix.setQuick(row, col, values[row][col]);
      }
    }
    return matrix;
  }

}
