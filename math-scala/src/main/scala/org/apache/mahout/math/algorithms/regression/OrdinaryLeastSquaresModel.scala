/**
  * Licensed to the Apache Software Foundation (ASF) under one
  * or more contributor license agreements. See the NOTICE file
  * distributed with this work for additional information
  * regarding copyright ownership. The ASF licenses this file
  * to you under the Apache License, Version 2.0 (the
  * "License"); you may not use this file except in compliance
  * with the License. You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing,
  * software distributed under the License is distributed on an
  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  * KIND, either express or implied. See the License for the
  * specific language governing permissions and limitations
  * under the License.
  */

package org.apache.mahout.math.algorithms.regression

import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._

class OrdinaryLeastSquaresModel[K]
  extends LinearRegressorModel[K] {
  // https://en.wikipedia.org/wiki/Ordinary_least_squares

  def predict(drmPredictors: DrmLike[K]): DrmLike[K] = {
    var X = drmPredictors
    if (addIntercept) {
      X = X cbind 1
    }
    X %*% beta
  }

}

class OrdinaryLeastSquares[K] extends LinearRegressorFitter[K] {


  def fit(drmFeatures: DrmLike[K],
          drmTarget: DrmLike[K],
          hyperparameters: (Symbol, Any)*): OrdinaryLeastSquaresModel[K] = {

    assert(drmTarget.ncol == 1, s"drmTarget must be a single column matrix, found ${drmTarget.ncol} columns")
    var model = new OrdinaryLeastSquaresModel[K]()
    setStandardHyperparameters(hyperparameters.toMap)


    if (drmFeatures.nrow != drmTarget.nrow){
      throw new Exception(s"${drmFeatures.nrow} observations in features, ${drmTarget.nrow} observations in target, must be equal.")
    }

    var X = drmFeatures

    if (addIntercept) {
      X = X cbind 1
    }

    val XtX = (X.t %*% X).collect
    val drmXtXinv = solve(XtX)
    val drmXty = (X.t %*% drmTarget).collect // this fails when number of columns^2 size matrix won't fit in driver
    model.beta = (drmXtXinv %*% drmXty)(::, 0)


    this.modelPostprocessing(model, X, drmTarget, drmXtXinv)
  }
}
