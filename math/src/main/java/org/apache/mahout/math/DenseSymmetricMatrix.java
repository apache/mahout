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

/**
 * Economy packaging for a dense symmetric in-core matrix.
 */
public class DenseSymmetricMatrix extends UpperTriangular {
  public DenseSymmetricMatrix(int n) {
    super(n);
  }

  public DenseSymmetricMatrix(double[] data, boolean shallow) {
    super(data, shallow);
  }

  public DenseSymmetricMatrix(Vector data) {
    super(data);
  }

  public DenseSymmetricMatrix(UpperTriangular mx) {
    super(mx);
  }

  @Override
  public double getQuick(int row, int column) {
    if (column < row) {
      int swap = row;
      row = column;
      column = swap;
    }
    return super.getQuick(row, column);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    if (column < row) {
      int swap = row;
      row = column;
      column = swap;
    }
    super.setQuick(row, column, value);
  }

}
