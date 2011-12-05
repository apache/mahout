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
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.util.Iterator;

/** see <a href="http://research.yahoo.com/pub/2433">Collaborative Filtering for Implicit Feedback Datasets</a> */
public class ImplicitFeedbackAlternatingLeastSquaresSolver {

  private final int numFeatures;
  private final double alpha;
  private final double lambda;

  private final OpenIntObjectHashMap<Vector> Y;
  private final Matrix YtransposeY;

  public ImplicitFeedbackAlternatingLeastSquaresSolver(int numFeatures, double lambda, double alpha,
      OpenIntObjectHashMap Y) {
    this.numFeatures = numFeatures;
    this.lambda = lambda;
    this.alpha = alpha;
    this.Y = Y;
    YtransposeY = YtransposeY(Y);
  }

  public Vector solve(Vector ratings) {
    return solve(YtransposeY.plus(YtransponseCuMinusIYPlusLambdaI(ratings)), YtransponseCuPu(ratings));
  }

  private static Vector solve(Matrix A, Matrix y) {
    return new QRDecomposition(A).solve(y).viewColumn(0);
  }

  protected double confidence(double rating) {
    return 1 + alpha * rating;
  }

  /* Y' Y */
  private Matrix YtransposeY(OpenIntObjectHashMap<Vector> Y) {

    Matrix compactedY = new DenseMatrix(Y.size(), numFeatures);
    IntArrayList indexes = Y.keys();
    indexes.quickSort();

    int row = 0;
    for (int index : indexes.elements()) {
      compactedY.assignRow(row++, Y.get(index));
    }

    return compactedY.transpose().times(compactedY);
  }

  /** Y' (Cu - I) Y + λ I */
  private Matrix YtransponseCuMinusIYPlusLambdaI(Vector userRatings) {
    Preconditions.checkArgument(userRatings.isSequentialAccess(), "need sequential access to ratings!");

    /* (Cu -I) Y */
    OpenIntObjectHashMap<Vector> CuMinusIY = new OpenIntObjectHashMap<Vector>();
    Iterator<Vector.Element> ratings = userRatings.iterateNonZero();
    while (ratings.hasNext()) {
      Vector.Element e = ratings.next();
      CuMinusIY.put(e.index(), Y.get(e.index()).times(confidence(e.get()) - 1));
    }

    Matrix YtransponseCuMinusIY = new DenseMatrix(numFeatures, numFeatures);

    /* Y' (Cu -I) Y by outer products */
    ratings = userRatings.iterateNonZero();
    while (ratings.hasNext()) {
      Vector.Element e = ratings.next();
      for (Vector.Element feature : Y.get(e.index())) {
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
  private Matrix YtransponseCuPu(Vector userRatings) {
    Preconditions.checkArgument(userRatings.isSequentialAccess(), "need sequential access to ratings!");

    Vector YtransponseCuPu = new DenseVector(numFeatures);

    Iterator<Vector.Element> ratings = userRatings.iterateNonZero();
    while (ratings.hasNext()) {
      Vector.Element e = ratings.next();
      YtransponseCuPu.assign(Y.get(e.index()).times(confidence(e.get())), Functions.PLUS);
    }

    return columnVectorAsMatrix(YtransponseCuPu);
  }

  private Matrix columnVectorAsMatrix(Vector v) {
    Matrix matrix = new DenseMatrix(numFeatures, 1);
    for (Vector.Element e : v) {
      matrix.setQuick(e.index(), 0, e.get());
    }
    return matrix;
  }

}
