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

import org.apache.mahout.math.function.Functions;
import org.junit.Test;

public class MatrixVectorViewTest extends MahoutTestCase {

  /**
   * Test for the error reported in https://issues.apache.org/jira/browse/MAHOUT-1146
   */
  @Test
  public void testColumnView() {

    Matrix matrix = new DenseMatrix(5, 3);
    Vector column2 = matrix.viewColumn(2);
    Matrix outerProduct = column2.cross(column2);

    assertEquals(matrix.numRows(), outerProduct.numRows());
    assertEquals(matrix.numRows(), outerProduct.numCols());
  }

  /**
   * Test for out of range column or row access.
   */
  @Test
  public void testIndexRange() {
    Matrix m = new DenseMatrix(20, 30).assign(Functions.random());
    try {
      m.viewColumn(30);
      fail("Should have thrown exception");
    } catch (IllegalArgumentException e) {
      assertTrue(e.getMessage().startsWith("Index 30 is outside allowable"));
    }
    try {
      m.viewRow(20);
      fail("Should have thrown exception");
    } catch (IllegalArgumentException e) {
      assertTrue(e.getMessage().startsWith("Index 20 is outside allowable"));
    }
  }
}
