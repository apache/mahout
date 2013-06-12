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

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

/**
 * Provides a way to get data from a file and treat it as if it were a matrix, but avoids putting
 * all that data onto the Java heap.  Instead, the file is mapped into non-heap memory as a
 * DoubleBuffer and we access that instead.  The interesting aspect of this is that the values in
 * the matrix are binary and sparse so we don't need to store the actual data, just the location of
 * non-zero values.
 * <p/>
 * Currently file data is formatted as follows:
 * <p/>
 * <ul> <li>A magic number to indicate the file format.</li> <li>The size of the matrix (max rows
 * and columns possible)</li> <li>Number of non-zeros in each row.</li> <li>A list of non-zero
 * columns for each row.  The list starts with a count and then has column numbers</li> </ul>
 * <p/>
 * It would be preferable to use something like protobufs to define the format so that we can use
 * different row formats for different kinds of data.  For instance, Golay coding of column numbers
 * or compressed bit vectors might be good representations for some purposes.
 */
public final class FileBasedSparseBinaryMatrix extends AbstractMatrix {
  private static final int MAGIC_NUMBER_V0 = 0x12d7067d;

  private final List<IntBuffer> data = Lists.newArrayList();
  private int[] bufferIndex;
  private int[] rowOffset;
  private int[] rowSize;

  /**
   * Constructs an empty matrix of the given size.
   *
   * @param rows    The number of rows in the result.
   * @param columns The number of columns in the result.
   */
  public FileBasedSparseBinaryMatrix(int rows, int columns) {
    super(rows, columns);
  }

  public void setData(File f) throws IOException {
    List<ByteBuffer> buffers = Lists.newArrayList();
    FileChannel input = new FileInputStream(f).getChannel();

    buffers.add(input.map(FileChannel.MapMode.READ_ONLY, 0, Math.min(Integer.MAX_VALUE, f.length())));
    data.add(buffers.get(0).asIntBuffer());
    Preconditions.checkArgument(buffers.get(0).getInt() == MAGIC_NUMBER_V0, "Wrong type of file");

    int rows = buffers.get(0).getInt();
    int cols = buffers.get(0).getInt();
    Preconditions.checkArgument(rows == rowSize());
    Preconditions.checkArgument(cols == columnSize());

    rowOffset = new int[rows];
    rowSize = new int[rows];
    bufferIndex = new int[rows];

    int offset = 12 + 4 * rows;
    for (int i = 0; i < rows; i++) {
      int size = buffers.get(0).getInt();
      int buffer = 0;
      while (buffer < buffers.size()) {
        if (offset + size * 4 <= buffers.get(buffer).limit()) {
          break;
        } else {
          offset -= buffers.get(buffer).capacity();
        }
      }
      if (buffer == buffers.size()) {
        buffers.add(input.map(FileChannel.MapMode.READ_ONLY, 0, Math.min(Integer.MAX_VALUE, f.length() - offset)));
        data.add(buffers.get(buffer).asIntBuffer());
      }
      rowOffset[i] = offset / 4;
      rowSize[i] = size;
      bufferIndex[i] = buffer;

//      final SparseBinaryVector v = new SparseBinaryVector(buffers.get(buffer), columns, offset, size);
//      this.rows.add(v);
      offset += size * 4;
    }
  }

  public static void writeMatrix(File f, Matrix m) throws IOException {
    Preconditions.checkArgument(f.canWrite(), "Can't write to output file");
    FileOutputStream fos = new FileOutputStream(f);

    // write header
    DataOutputStream out = new DataOutputStream(fos);
    out.writeInt(MAGIC_NUMBER_V0);
    out.writeInt(m.rowSize());
    out.writeInt(m.columnSize());

    // compute offsets and write row headers
    for (MatrixSlice row : m) {
      int nondefaultElements = row.vector().getNumNondefaultElements();
      out.writeInt(nondefaultElements);
    }

    // write rows
    for (MatrixSlice row : m) {
      List<Integer> columns = Lists.newArrayList(Iterables.transform(row.vector().nonZeroes(),
        new Function<Vector.Element, Integer>() {
          @Override
          public Integer apply(Vector.Element element) {
            return element.index();
          }
        }));
      Collections.sort(columns);

      for (Integer column : columns) {
        out.writeInt(column);
      }
    }

    out.close();
    fos.close();
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
   * @param rowIndex    an int row index
   * @param columnIndex an int column index
   * @return the double at the index
   */
  @Override
  public double getQuick(int rowIndex, int columnIndex) {
    IntBuffer tmp = data.get(bufferIndex[rowIndex]).asReadOnlyBuffer();
    tmp.position(rowOffset[rowIndex]);
    tmp.limit(rowSize[rowIndex]);
    tmp = tmp.slice();
    return searchForIndex(tmp, columnIndex);
  }

  private static double searchForIndex(IntBuffer row, int columnIndex) {
    int high = row.limit();
    if (high == 0) {
      return 0;
    }
    int low = 0;
    while (high > low) {
      int mid = (low + high) / 2;
      if (row.get(mid) < columnIndex) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    if (low >= row.limit()) {
      return 0;
    } else if (high == low && row.get(low) == columnIndex) {
      return 1;
    } else {
      return 0;
    }
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
   * Returns an empty matrix of the same underlying class as the receiver and of the specified
   * size.
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
   * Return a view into part of a matrix.  Changes to the view will change the original matrix.
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

  /**
   * Returns a view of a row.  Changes to the view will affect the original.
   *
   * @param rowIndex Which row to return.
   * @return A vector that references the desired row.
   */
  @Override
  public Vector viewRow(int rowIndex) {
    IntBuffer tmp = data.get(bufferIndex[rowIndex]).asReadOnlyBuffer();
    tmp.position(rowOffset[rowIndex]);
    tmp.limit(rowOffset[rowIndex] + rowSize[rowIndex]);
    tmp = tmp.slice();
    return new SparseBinaryVector(tmp, columnSize());
  }

  private static class SparseBinaryVector extends AbstractVector {
    private final IntBuffer buffer;
    private final int maxIndex;

    private SparseBinaryVector(IntBuffer buffer, int maxIndex) {
      super(maxIndex);
      this.buffer = buffer;
      this.maxIndex = maxIndex;
    }

    SparseBinaryVector(ByteBuffer row, int maxIndex, int offset, int size) {
      super(maxIndex);
      row = row.asReadOnlyBuffer();
      row.position(offset);
      row.limit(offset + size * 4);
      row = row.slice();
      this.buffer = row.slice().asIntBuffer();
      this.maxIndex = maxIndex;
    }

    /**
     * Subclasses must override to return an appropriately sparse or dense result
     *
     * @param rows    the row cardinality
     * @param columns the column cardinality
     * @return a Matrix
     */
    @Override
    protected Matrix matrixLike(int rows, int columns) {
      throw new UnsupportedOperationException("Default operation");
    }

    /**
     * Used internally by assign() to update multiple indices and values at once.
     * Only really useful for sparse vectors (especially SequentialAccessSparseVector).
     * <p/>
     * If someone ever adds a new type of sparse vectors, this method must merge (index, value) pairs into the vector.
     *
     * @param updates a mapping of indices to values to merge in the vector.
     */
    @Override
    public void mergeUpdates(OrderedIntDoubleMapping updates) {
      throw new UnsupportedOperationException("Cannot mutate SparseBinaryVector");
    }

    /**
     * @return true iff this implementation should be considered dense -- that it explicitly represents
     *         every value
     */
    @Override
    public boolean isDense() {
      return false;
    }

    /**
     * @return true iff this implementation should be considered to be iterable in index order in an
     *         efficient way. In particular this implies that {@link #iterator()} and {@link
     *         #iterateNonZero()} return elements in ascending order by index.
     */
    @Override
    public boolean isSequentialAccess() {
      return true;
    }

    /**
     * Iterates over all elements
     *
     * NOTE: Implementations may choose to reuse the Element returned
     * for performance reasons, so if you need a copy of it, you should call {@link #getElement(int)}
     * for the given index
     *
     * @return An {@link java.util.Iterator} over all elements
     */
    @Override
    public Iterator<Element> iterator() {
      return new AbstractIterator<Element>() {
        int i = 0;

        @Override
        protected Element computeNext() {
          if (i < maxIndex) {
            return new Element() {
              int index = i++;
              /**
               * @return the value of this vector element.
               */
              @Override
              public double get() {
                return getQuick(index);
              }

              /**
               * @return the index of this vector element.
               */
              @Override
              public int index() {
                return index;
              }

              /**
               * @param value Set the current element to value.
               */
              @Override
              public void set(double value) {
                throw new UnsupportedOperationException("Default operation");
              }
            };
          } else {
            return endOfData();
          }
        }
      };
    }

    /**
      * Iterates over all non-zero elements. <p/> NOTE: Implementations may choose to reuse the Element
      * returned for performance reasons, so if you need a copy of it, you should call {@link
      * #getElement(int)} for the given index
      *
      * @return An {@link java.util.Iterator} over all non-zero elements
      */
    @Override
    public Iterator<Element> iterateNonZero() {
      return new AbstractIterator<Element>() {
        int i = 0;
        @Override
        protected Element computeNext() {
          if (i < buffer.limit()) {
            return new BinaryReadOnlyElement(buffer.get(i++));
          } else {
            return endOfData();
          }
        }
      };
    }

  /**
     * Return the value at the given index, without checking bounds
     *
     * @param index an int index
     * @return the double at the index
     */
    @Override
    public double getQuick(int index) {
      return searchForIndex(buffer, index);
    }

    /**
     * Return an empty vector of the same underlying class as the receiver
     *
     * @return a Vector
     */
    @Override
    public Vector like() {
      return new RandomAccessSparseVector(size());
    }

    /**
     * Copy the vector for fast operations.
     *
     * @return a Vector
     */
    @Override
    protected Vector createOptimizedCopy() {
      return new RandomAccessSparseVector(size()).assign(this);
    }

    /**
     * Set the value at the given index, without checking bounds
     *
     * @param index an int index into the receiver
     * @param value a double value to set
     */
    @Override
    public void setQuick(int index, double value) {
      throw new UnsupportedOperationException("Read-only view");
    }

    /**
     * Set the value at the given index, without checking bounds
     *
     * @param index an int index into the receiver
     * @param increment a double value to set
     */
    @Override
    public void incrementQuick(int index, double increment) {
      throw new UnsupportedOperationException("Read-only view");
    }

    /**
     * Return the number of values in the recipient which are not the default value.  For instance, for
     * a sparse vector, this would be the number of non-zero values.
     *
     * @return an int
     */
    @Override
    public int getNumNondefaultElements() {
      return buffer.limit();
    }

    @Override
    public double getLookupCost() {
      return 1;
    }

    @Override
    public double getIteratorAdvanceCost() {
      return 1;
    }

    @Override
    public boolean isAddConstantTime() {
      throw new UnsupportedOperationException("Can't add binary value");
    }
  }

  public static class BinaryReadOnlyElement implements Vector.Element {
    private final int index;

    public BinaryReadOnlyElement(int index) {
      this.index = index;
    }

    /**
     * @return the value of this vector element.
     */
    @Override
    public double get() {
      return 1;
    }

    /**
     * @return the index of this vector element.
     */
    @Override
    public int index() {
      return index;
    }

    /**
     * @param value Set the current element to value.
     */
    @Override
    public void set(double value) {
      throw new UnsupportedOperationException("Can't set binary value");
    }
  }
}
