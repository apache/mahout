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

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Random matrix.  Each value is taken from {-1,0,1} with roughly equal probability.  Note
 * that by default, the value is determined by a relatively simple hash of the coordinates.
 * Such a hash is not usable where real randomness is required, but suffices nicely for
 * random projection methods.
 *
 * If the simple hash method is not satisfactory, an optional high quality mode is available
 * which uses a murmur hash of the coordinates.
 */
public class RandomTrinaryMatrix extends AbstractMatrix {
  private static final AtomicInteger ID = new AtomicInteger();
  private static final int PRIME1 = 104047;
  private static final int PRIME2 = 101377;
  private static final int PRIME3 = 64661;
  private static final long SCALE = 1L << 32;

  private final int seed;

  // set this to true to use a high quality hash
  private boolean highQuality = false;

  public RandomTrinaryMatrix(int seed, int rows, int columns, boolean highQuality) {
    super(rows, columns);

    this.highQuality = highQuality;
    this.seed = seed;
  }

  public RandomTrinaryMatrix(int rows, int columns) {
    this(ID.incrementAndGet(), rows, columns, false);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    throw new UnsupportedOperationException("Can't assign to read-only matrix");
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    throw new UnsupportedOperationException("Can't assign to read-only matrix");
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
    if (highQuality) {
      ByteBuffer buf = ByteBuffer.allocate(8);
      buf.putInt(row);
      buf.putInt(column);
      buf.flip();
      return (MurmurHash.hash64A(buf, seed) & (SCALE - 1)) / (double) SCALE;
    } else {
      // this isn't a fantastic random number generator, but it is just fine for random projections
      return ((((row * PRIME1) + column * PRIME2 + row * column * PRIME3) & 8) * 0.25) - 1;
    }
  }


  /**
   * Return an empty matrix of the same underlying class as the receiver
   *
   * @return a Matrix
   */
  @Override
  public Matrix like() {
    return new DenseMatrix(rowSize(), columnSize());
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
    throw new UnsupportedOperationException("Can't assign to read-only matrix");
  }

  /**
   * Return the number of values in the recipient
   *
   * @return an int[2] containing [row, column] count
   */
  @Override
  public int[] getNumNondefaultElements() {
    throw new UnsupportedOperationException("Can't assign to read-only matrix");
  }

  /**
   * Return a new matrix containing the subset of the recipient
   *
   * @param offset an int[2] offset into the receiver
   * @param size   the int[2] size of the desired result
   * @return a new Matrix that is a view of the original
   * @throws org.apache.mahout.math.CardinalityException
   *          if the length is greater than the cardinality of the receiver
   * @throws org.apache.mahout.math.IndexException
   *          if the offset is negative or the offset+length is outside of the receiver
   */
  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    return new MatrixView(this, offset, size);
  }
}
