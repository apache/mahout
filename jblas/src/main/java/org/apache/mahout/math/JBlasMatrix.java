/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.math;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.AbstractMatrix;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.SparseMatrix;

public class JBlasMatrix extends AbstractMatrix {

  public JBlasMatrix(int rows, int cols) {
    super(rows, cols);
  }


  public void setQuick(int row, int col, double val) {

  }

  public Matrix like(int nrow, int ncol) {
    return new JBlasMatrix(nrow, ncol);
  }

  public Matrix like() {
    return new JBlasMatrix(rowSize(), columnSize());
  }

  public double getQuick(int row, int col) {
    return 0.0;
  }

  public Matrix assignRow(int row, Vector v) {
    return null;
  }

  public Matrix assignColumn(int col, Vector v) {
    return null;
  }
}
