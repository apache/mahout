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

package org.apache.mahout.math.hadoop.stochasticsvd;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Varint;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.PlusMult;

/**
 * block that supports accumulating rows and their sums , suitable for combiner
 * and reducers of multiplication jobs.
 */
public class SparseRowBlockWritable implements Writable {

  private int[] rowIndices;
  private Vector[] rows;
  private int numRows;

  public SparseRowBlockWritable() {
    this(10);
  }

  public SparseRowBlockWritable(int initialCapacity) {
    rowIndices = new int[initialCapacity];
    rows = new Vector[initialCapacity];
  }

  public int[] getRowIndices() {
    return rowIndices;
  }

  public Vector[] getRows() {
    return rows;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    numRows = Varint.readUnsignedVarInt(in);
    if (rows == null || rows.length < numRows) {
      rows = new Vector[numRows];
      rowIndices = new int[numRows];
    }
    VectorWritable vw = new VectorWritable();
    for (int i = 0; i < numRows; i++) {
      rowIndices[i] = Varint.readUnsignedVarInt(in);
      vw.readFields(in);
      rows[i] = vw.get().clone();
    }

  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeUnsignedVarInt(numRows, out);
    VectorWritable vw = new VectorWritable();
    for (int i = 0; i < numRows; i++) {
      Varint.writeUnsignedVarInt(rowIndices[i], out);
      vw.set(rows[i]);
      vw.write(out);
    }
  }

  public void plusRow(int index, Vector row) {
    /*
     * often accumulation goes in row-increasing order, so check for this to
     * avoid binary search (another log Height multiplier).
     */

    int pos =
      numRows == 0 || rowIndices[numRows - 1] < index ? -numRows - 1 : Arrays
        .binarySearch(rowIndices, 0, numRows, index);
    if (pos >= 0) {
      rows[pos].assign(row, PlusMult.plusMult(1));
    } else {
      insertIntoPos(-pos - 1, index, row);
    }
  }

  private void insertIntoPos(int pos, int rowIndex, Vector row) {
    // reallocate if needed
    if (numRows == rows.length) {
      rows = Arrays.copyOf(rows, numRows + 1 << 1);
      rowIndices = Arrays.copyOf(rowIndices, numRows + 1 << 1);
    }
    // make a hole if needed
    System.arraycopy(rows, pos, rows, pos + 1, numRows - pos);
    System.arraycopy(rowIndices, pos, rowIndices, pos + 1, numRows - pos);
    // put
    rowIndices[pos] = rowIndex;
    rows[pos] = row.clone();
    numRows++;
  }

  /**
   * pluses one block into another. Use it for accumulation of partial products in
   * combiners and reducers.
   * 
   * @param bOther
   *          block to add
   */
  public void plusBlock(SparseRowBlockWritable bOther) {
    /*
     * since we maintained row indices in a sorted order, we can run sort merge
     * to expedite this operation
     */
    int i = 0;
    int j = 0;
    while (i < numRows && j < bOther.numRows) {
      while (i < numRows && rowIndices[i] < bOther.rowIndices[j]) {
        i++;
      }
      if (i < numRows) {
        if (rowIndices[i] == bOther.rowIndices[j]) {
          rows[i].assign(bOther.rows[j], PlusMult.plusMult(1));
        } else {
          // insert into i-th position
          insertIntoPos(i, bOther.rowIndices[j], bOther.rows[j]);
        }
        // increment in either case
        i++;
        j++;
      }
    }
    for (; j < bOther.numRows; j++) {
      insertIntoPos(numRows, bOther.rowIndices[j], bOther.rows[j]);
    }
  }

  public int getNumRows() {
    return numRows;
  }

  public void clear() {
    numRows = 0;
    Arrays.fill(rows, null);
  }
}
