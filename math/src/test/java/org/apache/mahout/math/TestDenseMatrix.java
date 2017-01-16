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

import org.junit.Assert;
import org.junit.Test;

public final class TestDenseMatrix extends MatrixTest {

  @Override
  public Matrix matrixFactory(double[][] values) {
    return new DenseMatrix(values);
  }

  @Test
  public void testGetValues() {
    DenseMatrix m = new DenseMatrix(10, 10);
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        m.set(i, j, 10 * i + j);
      }
    }

    double[][] values = m.getBackingStructure();
    Assert.assertEquals(values.length, 10);
    Assert.assertEquals(values[0].length, 10);
    Assert.assertEquals(values[9][9], 99.0, 0.0);
  }

}
