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
import java.util.Map;

public class MatrixWritable implements Writable {

  private Matrix matrix;

  public Matrix get() { return matrix; }

  public void set(Matrix matrix) {
    this.matrix = matrix;
  }

  public MatrixWritable() {

  }

  public MatrixWritable(Matrix m) {
    set(m);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    writeMatrix(out, matrix);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    matrix = readMatrix(in);
  }

  public static void readLabels(DataInput in,
                                Map<String, Integer> columnLabelBindings,
                                Map<String, Integer> rowLabelBindings) throws IOException {
    int colSize = in.readInt();
    if (colSize > 0) {
      for (int i = 0; i < colSize; i++) {
        columnLabelBindings.put(in.readUTF(), in.readInt());
      }
    }
    int rowSize = in.readInt();
    if (rowSize > 0) {
      rowLabelBindings = new HashMap<String, Integer>();
      for (int i = 0; i < rowSize; i++) {
        rowLabelBindings.put(in.readUTF(), in.readInt());
      }
    }
  }

  public static void writeLabelBindings(DataOutput out,
                                        Map<String, Integer> columnLabelBindings,
                                        Map<String, Integer> rowLabelBindings) throws IOException {
    if (columnLabelBindings == null) {
      out.writeInt(0);
    } else {
      out.writeInt(columnLabelBindings.size());
      for (Map.Entry<String, Integer> stringIntegerEntry : columnLabelBindings.entrySet()) {
        out.writeUTF(stringIntegerEntry.getKey());
        out.writeInt(stringIntegerEntry.getValue());
      }
    }
    if (rowLabelBindings == null) {
      out.writeInt(0);
    } else {
      out.writeInt(rowLabelBindings.size());
      for (Map.Entry<String, Integer> stringIntegerEntry : rowLabelBindings.entrySet()) {
        out.writeUTF(stringIntegerEntry.getKey());
        out.writeInt(stringIntegerEntry.getValue());
      }
    }
  }

  /** Reads a typed Matrix instance from the input stream */
  public static Matrix readMatrix(DataInput in) throws IOException {
    String matrixClassName = in.readUTF();
    Matrix matrix;
    try {
      matrix = Class.forName(matrixClassName).asSubclass(Matrix.class)
          .newInstance();
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
   // matrix.readFields(in);
    return matrix;
  }

  /** Writes a typed Matrix instance to the output stream */
  public static void writeMatrix(DataOutput out, Matrix matrix)
      throws IOException {
    out.writeUTF(matrix.getClass().getName());
   // matrix.write(out);
  }
}
