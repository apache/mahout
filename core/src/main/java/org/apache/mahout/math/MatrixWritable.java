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

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Map;

public class MatrixWritable implements Writable {

  private static final int FLAG_DENSE = 0x01;
  private static final int FLAG_SEQUENTIAL = 0x02;
  private static final int FLAG_LABELS = 0x04;
  private static final int NUM_FLAGS = 3;

  private Matrix matrix;

  public MatrixWritable() {
  }

  public MatrixWritable(Matrix m) {
    set(m);
  }

  public Matrix get() {
    return matrix;
  }

  public void set(Matrix matrix) {
    this.matrix = matrix;
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
    int flags = in.readInt();
    Preconditions.checkArgument(flags >> NUM_FLAGS == 0, "Unknown flags set: %d", Integer.toString(flags, 2));
    boolean dense = (flags & FLAG_DENSE) != 0;
    boolean sequential = (flags & FLAG_SEQUENTIAL) != 0;
    boolean hasLabels = (flags & FLAG_LABELS) != 0;

    int rows = in.readInt();
    int columns = in.readInt();

    Matrix r;
    if (dense) {
      r = new DenseMatrix(rows, columns);
    } else {
      r = new SparseRowMatrix(rows, columns, !sequential);
    }

    for (int row = 0; row < rows; row++) {
      r.viewRow(row).assign(VectorWritable.readVector(in));
    }

    if (hasLabels) {
      Map<String,Integer> columnLabelBindings = Maps.newHashMap();
      Map<String,Integer> rowLabelBindings = Maps.newHashMap();
      readLabels(in, columnLabelBindings, rowLabelBindings);
      if (!columnLabelBindings.isEmpty()) {
        r.setColumnLabelBindings(columnLabelBindings);
      }
      if (!rowLabelBindings.isEmpty()) {
        r.setRowLabelBindings(rowLabelBindings);
      }
    }

    return r;
  }

  /** Writes a typed Matrix instance to the output stream */
  public static void writeMatrix(DataOutput out, Matrix matrix) throws IOException {
    int flags = 0;
    Vector row = matrix.viewRow(0);
    if (row.isDense()) {
      flags |= FLAG_DENSE;
    }
    if (row.isSequentialAccess()) {
      flags |= FLAG_SEQUENTIAL;
    }
    if (matrix.getRowLabelBindings() != null || matrix.getColumnLabelBindings() != null) {
      flags |= FLAG_LABELS;
    }
    out.writeInt(flags);

    out.writeInt(matrix.rowSize());
    out.writeInt(matrix.columnSize());

    for (int i = 0; i < matrix.rowSize(); i++) {
      VectorWritable.writeVector(out, matrix.viewRow(i), false);
    }
    if ((flags & FLAG_LABELS) != 0) {
      writeLabelBindings(out, matrix.getColumnLabelBindings(), matrix.getRowLabelBindings());
    }
  }
}
