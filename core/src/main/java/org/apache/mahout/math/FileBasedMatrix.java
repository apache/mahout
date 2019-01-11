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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

/**
 * Provides a way to get data from a file and treat it as if it were a matrix, but avoids putting all that
 * data onto the Java heap.  Instead, the file is mapped into non-heap memory as a DoubleBuffer and we access
 * that instead.
 */
public final class FileBasedMatrix extends AbstractMatrix {
  private final int rowsPerBlock;
  private final List<DoubleBuffer> content = Lists.newArrayList();

  /**
   * Constructs an empty matrix of the given size.
   *
   * @param rows    The number of rows in the result.
   * @param columns The number of columns in the result.
   */
  public FileBasedMatrix(int rows, int columns) {
    super(rows, columns);
    long maxRows = ((1L << 31) - 1) / (columns * 8);
    if (rows > maxRows) {
      rowsPerBlock = (int) maxRows;
    } else {
      rowsPerBlock = rows;
    }
  }

  private void addData(DoubleBuffer content) {
    this.content.add(content);
  }

  public void setData(File f, boolean loadNow) throws IOException {
    Preconditions.checkArgument(f.length() == rows * columns * 8L, "File " + f + " is wrong length");

    for (int i = 0; i < (rows + rowsPerBlock - 1) / rowsPerBlock; i++) {
      long start = i * rowsPerBlock * columns * 8L;
      long size = rowsPerBlock * columns * 8L;
      MappedByteBuffer buf = new FileInputStream(f).getChannel().map(FileChannel.MapMode.READ_ONLY, start,
                                                                     Math.min(f.length() - start, size));
      if (loadNow) {
        buf.load();
      }
      addData(buf.asDoubleBuffer());
    }
  }

  public static void writeMatrix(File f, Matrix m) throws IOException {
    Preconditions.checkArgument(f.canWrite(), "Can't write to output file");
    FileOutputStream fos = new FileOutputStream(f);
    try {
      ByteBuffer buf = ByteBuffer.allocate(m.columnSize() * 8);
      for (MatrixSlice row : m) {
        buf.clear();
        for (Vector.Element element : row.vector().all()) {
          buf.putDouble(element.get());
        }
        buf.flip();
        fos.write(buf.array());
      }
    } finally {
      fos.close();
    }
  }

  /**
   * Assign the other vector values to the column of the receiver
   *
   * @param column the int row to assign
   * @param other  a Vector
   * @return the modified receiver
   * @throws org.apache.mahout.math.CardinalityException
   *          if the cardinalities differ
   */
  @Override
  public Matrix assignColumn(int column, Vector other) {
    throw new UnsupportedOperationException("Default operation");
  }

  /**
   * Assign the other vector values to the row of the receiver
   *
   * @param row   the int row to assign
   * @param other a Vector
   * @return the modified receiver
   * @throws org.apache.mahout.math.CardinalityException
   *          if the cardinalities differ
   */
  @Override
  public Matrix assignRow(int row, Vector other) {
    throw new UnsupportedOperationException("Default operation");
  }

  /**
   * Return the value at the given indexes, without checking bounds
   *
   * @param row    an int row index
   * @param column an int column index
   * @return the double at the index
   */
  @Override
  public double getQuick(int row, int column) {
    int block = row / rowsPerBlock;
    return content.get(block).get((row % rowsPerBlock) * columns + column);
  }

  /**
   * Return an empty matrix of the same underlying class as the receiver
   *
   * @return a Matrix
   */
  @Override
  public Matrix like() {
    throw new UnsupportedOperationException("Default operation");
  }

  /**
   * Returns an empty matrix of the same underlying class as the receiver and of the specified size.
   *
   * @param rows    the int number of rows
   * @param columns the int number of columns
   */
  @Override
  public Matrix like(int rows, int columns) {
    return new DenseMatrix(rows, columns);
  }

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param row    an int row index into the receiver
   * @param column an int column index into the receiver
   * @param value  a double value to set
   */
  @Override
  public void setQuick(int row, int column, double value) {
    throw new UnsupportedOperationException("Default operation");
  }

  /**
   * Return a view into part of a matrix.  Changes to the view will change the
   * original matrix.
   *
   * @param offset an int[2] offset into the receiver
   * @param size   the int[2] size of the desired result
   * @return a matrix that shares storage with part of the original matrix.
   * @throws org.apache.mahout.math.CardinalityException
   *          if the length is greater than the cardinality of the receiver
   * @throws org.apache.mahout.math.IndexException
   *          if the offset is negative or the offset+length is outside of the receiver
   */
  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    throw new UnsupportedOperationException("Default operation");
  }
}
