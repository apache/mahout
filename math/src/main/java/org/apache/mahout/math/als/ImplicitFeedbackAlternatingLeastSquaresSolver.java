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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/** see <a href="http://research.yahoo.com/pub/2433">Collaborative Filtering for Implicit Feedback Datasets</a> */
public class ImplicitFeedbackAlternatingLeastSquaresSolver {

  private final int numFeatures;
  private final double alpha;
  private final double lambda;
  private final int numTrainingThreads;

  private final OpenIntObjectHashMap<Vector> Y;
  private final Matrix YtransposeY;
  
  private static final Logger log = LoggerFactory.getLogger(ImplicitFeedbackAlternatingLeastSquaresSolver.class);
  
  public ImplicitFeedbackAlternatingLeastSquaresSolver(int numFeatures, double lambda, double alpha,
      OpenIntObjectHashMap<Vector> Y, int numTrainingThreads) {
    this.numFeatures = numFeatures;
    this.lambda = lambda;
    this.alpha = alpha;
    this.Y = Y;
    this.numTrainingThreads = numTrainingThreads;
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
  public Matrix getYtransposeY(final OpenIntObjectHashMap<Vector> Y) {

    ExecutorService queue = Executors.newFixedThreadPool(numTrainingThreads);
    if (log.isInfoEnabled()) {
      log.info("Starting the computation of Y'Y");
    }
    long startTime = System.nanoTime();
    final IntArrayList indexes = Y.keys();
    final int numIndexes = indexes.size();
  
    final double[][] YtY = new double[numFeatures][numFeatures];
  
    // Compute Y'Y by dot products between the 'columns' of Y
    for (int i = 0; i < numFeatures; i++) {
      for (int j = i; j < numFeatures; j++) {
  
        final int ii = i;
        final int jj = j;
        queue.execute(new Runnable() {
          @Override
          public void run() {
            double dot = 0;
            for (int k = 0; k < numIndexes; k++) {
              Vector row = Y.get(indexes.getQuick(k));
              dot += row.getQuick(ii) * row.getQuick(jj);
            }
            YtY[ii][jj] = dot;
            if (ii != jj) {
              YtY[jj][ii] = dot;
            }
          }
        });
  
      }
    }
    queue.shutdown();
    try {
      queue.awaitTermination(1, TimeUnit.DAYS);
    } catch (InterruptedException e) {
      log.error("Error during Y'Y queue shutdown", e);
      throw new RuntimeException("Error during Y'Y queue shutdown");
    }
    if (log.isInfoEnabled()) {
      log.info("Computed Y'Y in " + (System.nanoTime() - startTime) / 1000000.0 + " ms" );
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
