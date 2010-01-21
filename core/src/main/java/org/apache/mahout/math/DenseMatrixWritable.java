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

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;


public class DenseMatrixWritable extends DenseMatrix implements Writable {

  @Override
  public void readFields(DataInput in) throws IOException {
    columnLabelBindings = new HashMap<String, Integer>();
    rowLabelBindings = new HashMap<String, Integer>();
    MatrixWritable.readLabels(in, columnLabelBindings, rowLabelBindings);
    int rows = in.readInt();
    int columns = in.readInt();
    this.values = new double[rows][columns];
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        this.values[row][column] = in.readDouble();
      }
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    MatrixWritable.writeLabelBindings(out, columnLabelBindings, rowLabelBindings);
    out.writeInt(rowSize());
    out.writeInt(columnSize());
    for (double[] row : values) {
      for (double value : row) {
        out.writeDouble(value);
      }
    }
  }

}
