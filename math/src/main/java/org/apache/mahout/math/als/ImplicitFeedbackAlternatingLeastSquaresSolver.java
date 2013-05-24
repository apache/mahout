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
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

/** see <a href="http://research.yahoo.com/pub/2433">Collaborative Filtering for Implicit Feedback Datasets</a> */
public class ImplicitFeedbackAlternatingLeastSquaresSolver {

  private final int numFeatures;
  private final double alpha;
  private final double lambda;

  private final OpenIntObjectHashMap<Vector> Y;
  private final Matrix YtransposeY;

  public ImplicitFeedbackAlternatingLeastSquaresSolver(int numFeatures, double lambda, double alpha,
      OpenIntObjectHashMap<Vector> Y) {
    this.numFeatures = numFeatures;
    this.lambda = lambda;
    this.alpha = alpha;
    this.Y = Y;
    YtransposeY = getYtransposeY(Y);
  }

  public Vector solve(Vector ratings) {
    return solve(YtransposeY.plus(getYtransponseCuMinusIYPlusLambdaI(ratings)), getYtransponseCuPu(ratings));
  }

  private static Vector solve(Matrix A, Matrix y) {
    return new QRDecomposition(A).solve(y).viewColumn(0);
  }

  double confidence(double rating) {
    return 1 + alpha * rating;
  }

  /* Y' Y */
  private Matrix getYtransposeY(OpenIntObjectHashMap<Vector> Y) {

    IntArrayList indexes = Y.keys();
    indexes.quickSort();
    int numIndexes = indexes.size();

    double[][] YtY = new double[numFeatures][numFeatures];

    // Compute Y'Y by dot products between the 'columns' of Y
    for (int i = 0; i < numFeatures; i++) {
      for (int j = i; j < numFeatures; j++) {
        double dot = 0;
        for (int k = 0; k < numIndexes; k++) {
          Vector row = Y.get(indexes.getQuick(k));
          dot += row.getQuick(i) * row.getQuick(j);
        }
        YtY[i][j] = dot;
        if (i != j) {
          YtY[j][i] = dot;
        }
      }
    }
    return new DenseMatrix(YtY, true);
  }

  /** Y' (Cu - I) Y + λ I */
  private Matrix getYtransponseCuMinusIYPlusLambdaI(Vector userRatings) {
    Preconditions.checkArgument(userRatings.isSequentialAccess(), "need sequential access to ratings!");

    /* (Cu -I) Y */
    OpenIntObjectHashMap<Vector> CuMinusIY = new OpenIntObjectHashMap<Vector>(userRatings.getNumNondefaultElements());
    for (Element e : userRatings.nonZeroes()) {
      CuMinusIY.put(e.index(), Y.get(e.index()).times(confidence(e.get()) - 1));
    }

    Matrix YtransponseCuMinusIY = new DenseMatrix(numFeatures, numFeatures);

    /* Y' (Cu -I) Y by outer products */
    for (Element e : userRatings.nonZeroes()) {
      for (Vector.Element feature : Y.get(e.index()).all()) {
        Vector partial = CuMinusIY.get(e.index()).times(feature.get());
        YtransponseCuMinusIY.viewRow(feature.index()).assign(partial, Functions.PLUS);
      }
    }

    /* Y' (Cu - I) Y + λ I  add lambda on the diagonal */
    for (int feature = 0; feature < numFeatures; feature++) {
      YtransponseCuMinusIY.setQuick(feature, feature, YtransponseCuMinusIY.getQuick(feature, feature) + lambda);
    }

    return YtransponseCuMinusIY;
  }

  /** Y' Cu p(u) */
  private Matrix getYtransponseCuPu(Vector userRatings) {
    Preconditions.checkArgument(userRatings.isSequentialAccess(), "need sequential access to ratings!");

    Vector YtransponseCuPu = new DenseVector(numFeatures);

    for (Element e : userRatings.nonZeroes()) {
      YtransponseCuPu.assign(Y.get(e.index()).times(confidence(e.get())), Functions.PLUS);
    }

    return columnVectorAsMatrix(YtransponseCuPu);
  }

  private Matrix columnVectorAsMatrix(Vector v) {
    double[][] matrix =  new double[numFeatures][1];
    for (Vector.Element e : v.all()) {
      matrix[e.index()][0] =  e.get();
    }
    return new DenseMatrix(matrix, true);
  }

}
