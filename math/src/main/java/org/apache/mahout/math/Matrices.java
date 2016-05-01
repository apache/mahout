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
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.IntIntFunction;

import java.util.Random;

public final class Matrices {

  /**
   * Create a matrix view based on a function generator.
   * <p/>
   * The generator needs to be idempotent, i.e. returning same value
   * for each combination of (row, column) argument sent to generator's
   * {@link IntIntFunction#apply(int, int)} call.
   *
   * @param rows      Number of rows in a view
   * @param columns   Number of columns in a view.
   * @param gf        view generator
   * @param denseLike type of matrix returne dby {@link org.apache.mahout.math.Matrix#like()}.
   * @return new matrix view.
   */
  public static Matrix functionalMatrixView(final int rows,
                                                  final int columns,
                                                  final IntIntFunction gf,
                                                  final boolean denseLike) {
    return new FunctionalMatrixView(rows, columns, gf, denseLike);
  }

  /**
   * Shorter form of {@link Matrices#functionalMatrixView(int, int,
   * org.apache.mahout.math.function.IntIntFunction, boolean)}.
   */
  public static Matrix functionalMatrixView(final int rows,
                                                  final int columns,
                                                  final IntIntFunction gf) {
    return new FunctionalMatrixView(rows, columns, gf);
  }

  /**
   * A read-only transposed view of a matrix argument.
   *
   * @param m original matrix
   * @return transposed view of original matrix
   */
  public static Matrix transposedView(final Matrix m) {

    Preconditions.checkArgument(!(m instanceof SparseColumnMatrix));

    if (m instanceof TransposedMatrixView) {
      return ((TransposedMatrixView) m).getDelegate();
    } else {
      return new TransposedMatrixView(m);
    }
  }

  /**
   * Random Gaussian matrix view.
   *
   * @param seed generator seed
   */
  public static Matrix gaussianView(final int rows,
                                          final int columns,
                                          long seed) {
    return functionalMatrixView(rows, columns, gaussianGenerator(seed), true);
  }


  /**
   * Matrix view based on uniform [-1,1) distribution.
   *
   * @param seed generator seed
   */
  public static Matrix symmetricUniformView(final int rows,
                                                  final int columns,
                                                  int seed) {
    return functionalMatrixView(rows, columns, uniformSymmetricGenerator(seed), true);
  }

  /**
   * Matrix view based on uniform [0,1) distribution.
   *
   * @param seed generator seed
   */
  public static Matrix uniformView(final int rows,
                                         final int columns,
                                         int seed) {
    return functionalMatrixView(rows, columns, uniformGenerator(seed), true);
  }

  /**
   * Generator for a matrix populated by random Gauissian values (Gaussian matrix view)
   *
   * @param seed The seed for the matrix.
   * @return Gaussian {@link IntIntFunction} generating matrix view with normal values
   */
  public static IntIntFunction gaussianGenerator(final long seed) {
    final Random rnd = RandomUtils.getRandom(seed);
    return new IntIntFunction() {
      @Override
      public double apply(int first, int second) {
        rnd.setSeed(seed ^ (((long) first << 32) | (second & 0xffffffffL)));
        return rnd.nextGaussian();
      }
    };
  }

  private static final double UNIFORM_DIVISOR = Math.pow(2.0, 64);

  /**
   * Uniform [-1,1) matrix generator function.
   * <p/>
   * WARNING: to keep things performant, it is stateful and so not thread-safe.
   * You'd need to create a copy per thread (with same seed) if shared between threads.
   *
   * @param seed - random seed initializer
   * @return Uniform {@link IntIntFunction} generator
   */
  public static IntIntFunction uniformSymmetricGenerator(final int seed) {
    return new IntIntFunction() {
      private byte[] data = new byte[8];

      @Override
      public double apply(int row, int column) {
        long d = ((long) row << Integer.SIZE) | (column & 0xffffffffL);
        for (int i = 0; i < 8; i++, d >>>= 8) data[i] = (byte) d;
        long hash = MurmurHash.hash64A(data, seed);
        return hash / UNIFORM_DIVISOR;
      }
    };
  }

  /**
   * Uniform [0,1) matrix generator function
   *
   * @param seed generator seed
   */
  public static IntIntFunction uniformGenerator(final int seed) {
    return Functions.chain(new DoubleFunction() {
      @Override
      public double apply(double x) {
        return (x + 1.0) / 2.0;
      }
    }, uniformSymmetricGenerator(seed));
  }

}
