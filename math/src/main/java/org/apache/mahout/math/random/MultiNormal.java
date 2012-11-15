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

package org.apache.mahout.math.random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;

import java.util.Random;

/**
 * Samples from a multi-variate normal distribution.
 * <p/>
 * This is done by sampling from several independent unit normal distributions to get a vector u.
 * The sample value that is returned is then A u + m where A is derived from the covariance matrix
 * and m is the mean of the result.
 * <p/>
 * If \Sigma is the desired covariance matrix, then you can use any value of A such that A' A =
 * \Sigma.  The Cholesky decomposition can be used to compute A if \Sigma is positive definite.
 * Slightly more expensive is to use the SVD U S V' = \Sigma and then set A = U \sqrt{S}.
 *
 * Useful special cases occur when \Sigma is diagonal so that A = \sqrt(\Sigma) or where \Sigma = r I.
 *
 * Another special case is where m = 0.
 */
public class MultiNormal implements Sampler<Vector> {
  private final Random gen;
  private final int dimension;
  private final Matrix scale;
  private final Vector mean;

  /**
   * Constructs a sampler with diagonal scale matrix.
   * @param diagonal The diagonal elements of the scale matrix.
   */
  public MultiNormal(Vector diagonal) {
    this(new DiagonalMatrix(diagonal), null);
  }

  /**
   * Constructs a sampler with diagonal scale matrix and (potentially)
   * non-zero mean.
   * @param diagonal The scale matrix's principal diagonal.
   * @param mean The desired mean.  Set to null if zero mean is desired.
   */
  public MultiNormal(Vector diagonal, Vector mean) {
    this(new DiagonalMatrix(diagonal), mean);
  }

  /**
   * Constructs a sampler with non-trivial scale matrix and mean.
   */
  public MultiNormal(Matrix a, Vector mean) {
    this(a, mean, a.columnSize());
  }

  public MultiNormal(int dimension) {
    this(null, null, dimension);
  }

  public MultiNormal(double radius, Vector mean) {
    this(new DiagonalMatrix(radius, mean.size()), mean);
  }

  private MultiNormal(Matrix scale, Vector mean, int dimension) {
    gen = RandomUtils.getRandom();
    this.dimension = dimension;
    this.scale = scale;
    this.mean = mean;
  }

  @Override
  public Vector sample() {
    Vector v = new DenseVector(dimension).assign(
      new DoubleFunction() {
        @Override
        public double apply(double ignored) {
          return gen.nextGaussian();
        }
      }
    );
    if (mean != null) {
      if (scale != null) {
        return scale.times(v).plus(mean);
      } else {
        return v.plus(mean);
      }
    } else {
      if (scale != null) {
        return scale.times(v);
      } else {
        return v;
      }
    }
  }

  public Vector getScale() {
    return mean;
  }
}
