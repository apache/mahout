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

package org.apache.mahout.math.als;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.Vector;

/**
 * See
 * <a href="http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf">
 * this paper.</a>
 */
public final class AlternatingLeastSquaresSolver {

  private AlternatingLeastSquaresSolver() {}

  //TODO make feature vectors a simple array
  public static Vector solve(Iterable<Vector> featureVectors, Vector ratingVector, double lambda, int numFeatures) {

    Preconditions.checkNotNull(featureVectors, "Feature Vectors cannot be null");
    Preconditions.checkArgument(!Iterables.isEmpty(featureVectors));
    Preconditions.checkNotNull(ratingVector, "Rating Vector cannot be null");
    Preconditions.checkArgument(ratingVector.getNumNondefaultElements() > 0, "Rating Vector cannot be empty");
    Preconditions.checkArgument(Iterables.size(featureVectors) == ratingVector.getNumNondefaultElements());

    int nui = ratingVector.getNumNondefaultElements();

    Matrix MiIi = createMiIi(featureVectors, numFeatures);
    Matrix RiIiMaybeTransposed = createRiIiMaybeTransposed(ratingVector);

    /* compute Ai = MiIi * t(MiIi) + lambda * nui * E */
    Matrix Ai = miTimesMiTransposePlusLambdaTimesNuiTimesE(MiIi, lambda, nui);
    /* compute Vi = MiIi * t(R(i,Ii)) */
    Matrix Vi = MiIi.times(RiIiMaybeTransposed);
    /* compute Ai * ui = Vi */
    return solve(Ai, Vi);
  }

  private static Vector solve(Matrix Ai, Matrix Vi) {
    return new QRDecomposition(Ai).solve(Vi).viewColumn(0);
  }

  static Matrix addLambdaTimesNuiTimesE(Matrix matrix, double lambda, int nui) {
    Preconditions.checkArgument(matrix.numCols() == matrix.numRows(), "Must be a Square Matrix");
    double lambdaTimesNui = lambda * nui;
    int numCols = matrix.numCols();
    for (int n = 0; n < numCols; n++) {
      matrix.setQuick(n, n, matrix.getQuick(n, n) + lambdaTimesNui);
    }
    return matrix;
  }

  private static Matrix miTimesMiTransposePlusLambdaTimesNuiTimesE(Matrix MiIi, double lambda, int nui) {

    double lambdaTimesNui = lambda * nui;
    int rows = MiIi.numRows();

    double[][] result = new double[rows][rows];

    for (int i = 0; i < rows; i++) {
      for (int j = i; j < rows; j++) {
        double dot = MiIi.viewRow(i).dot(MiIi.viewRow(j));
        if (i != j) {
          result[i][j] = dot;
          result[j][i] = dot;
        } else {
          result[i][i] = dot + lambdaTimesNui;
        }
      }
    }
    return new DenseMatrix(result, true);
  }


  static Matrix createMiIi(Iterable<Vector> featureVectors, int numFeatures) {
    double[][] MiIi =  new double[numFeatures][Iterables.size(featureVectors)];
    int n = 0;
    for (Vector featureVector : featureVectors) {
      for (int m = 0; m < numFeatures; m++) {
        MiIi[m][n] = featureVector.getQuick(m);
      }
      n++;
    }
    return new DenseMatrix(MiIi, true);
  }

  static Matrix createRiIiMaybeTransposed(Vector ratingVector) {
    Preconditions.checkArgument(ratingVector.isSequentialAccess(), "Ratings should be iterable in Index or Sequential Order");

    double[][] RiIiMaybeTransposed = new double[ratingVector.getNumNondefaultElements()][1];
    int index = 0;
    for (Vector.Element elem : ratingVector.nonZeroes()) {
      RiIiMaybeTransposed[index++][0] = elem.get();
    }
    return new DenseMatrix(RiIiMaybeTransposed, true);
  }
}
