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

import java.util.List;

import com.google.common.collect.Lists;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;

public final class RandomProjector {
  private RandomProjector() {
  }

  /**
   * Generates a basis matrix of size projectedVectorSize x vectorSize. Multiplying a a vector by
   * this matrix results in the projected vector.
   *
   * The rows of the matrix are sampled from a multi normal distribution.
   *
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a projection matrix
   */
  public static Matrix generateBasisNormal(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    basisMatrix.assign(new Normal());
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  /**
   * Generates a basis matrix of size projectedVectorSize x vectorSize. Multiplying a a vector by
   * this matrix results in the projected vector.
   *
   * The rows of a matrix are sample from a distribution where:
   * - +1 has probability 1/2,
   * - -1 has probability 1/2
   *
   * See Achlioptas, D. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.
   * Journal of Computer and System Sciences, 66(4), 671–687. doi:10.1016/S0022-0000(03)00025-4
   *
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a projection matrix
   */
  public static Matrix generateBasisPlusMinusOne(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    for (int i = 0; i < projectedVectorSize; ++i) {
      for (int j = 0; j < vectorSize; ++j) {
        basisMatrix.set(i, j, RandomUtils.nextInt(2) == 0 ? 1 : -1);
      }
    }
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  /**
   * Generates a basis matrix of size projectedVectorSize x vectorSize. Multiplying a a vector by
   * this matrix results in the projected vector.
   *
   * The rows of a matrix are sample from a distribution where:
   * - 0 has probability 2/3,
   * - +1 has probability 1/6,
   * - -1 has probability 1/6
   *
   * See Achlioptas, D. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.
   * Journal of Computer and System Sciences, 66(4), 671–687. doi:10.1016/S0022-0000(03)00025-4
   *
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a projection matrix
   */
  public static Matrix generateBasisZeroPlusMinusOne(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    Multinomial<Double> choice = new Multinomial<Double>();
    choice.add(0.0, 2 / 3.0);
    choice.add(Math.sqrt(3.0), 1 / 6.0);
    choice.add(-Math.sqrt(3.0), 1 / 6.0);
    for (int i = 0; i < projectedVectorSize; ++i) {
      for (int j = 0; j < vectorSize; ++j) {
        basisMatrix.set(i, j, choice.sample());
      }
    }
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  /**
   * Generates a list of projectedVectorSize vectors, each of size vectorSize. This looks like a
   * matrix of size (projectedVectorSize, vectorSize).
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a list of projection vectors
   */
  public static List<Vector> generateVectorBasis(int projectedVectorSize, int vectorSize) {
    DoubleFunction random = new Normal();
    List<Vector> basisVectors = Lists.newArrayList();
    for (int i = 0; i < projectedVectorSize; ++i) {
      Vector basisVector = new DenseVector(vectorSize);
      basisVector.assign(random);
      basisVector.normalize();
      basisVectors.add(basisVector);
    }
    return basisVectors;
  }
}
