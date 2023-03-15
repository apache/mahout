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

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._




class RidgeRegressionModel[K] extends LinearRegressorModel[K]{

  def predict(drmPredictors: DrmLike[K]): DrmLike[K] = {

    var X = drmPredictors
    if (addIntercept) {
      X = X cbind 1
    }
    X %*% beta
  }
}

class RidgeRegression[K] extends LinearRegressorFitter[K] {

  var lambda: Double = _

  override def setStandardHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    lambda = hyperparameters.asInstanceOf[Map[Symbol, Double]].getOrElse('lambda,1.0)
  }

  def fit(drmFeatures: DrmLike[K], drmTarget: DrmLike[K], hyperparameters: (Symbol, Any)*): RidgeRegressionModel[K] = {//lambda: Double = 0.1) = {


    var model = new RidgeRegressionModel[K]
    setStandardHyperparameters(hyperparameters.toMap)

    var X = drmFeatures

    if (addIntercept) {
      X = X cbind 1
    }

    val XTX = (X.t %*% X).collect
    val drmXtXinv = solve(XTX)
    val XTy = X.t %*% drmTarget
    val reg = diag(lambda, XTX.ncol)

    val sol = solve(XTX.plus(reg), XTy)

    model.beta = sol(::, 0)

    this.modelPostprocessing(model, X, drmTarget, drmXtXinv)

    }
}

