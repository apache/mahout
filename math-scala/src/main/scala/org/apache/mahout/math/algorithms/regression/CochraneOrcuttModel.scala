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

import org.apache.mahout.math.{Vector => MahoutVector}
import org.apache.mahout.math.drm.{CacheHint, DrmLike, safeToNonNegInt}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.RLikeOps._

class CochraneOrcuttModel[K](regressor: LinearRegressorModel[K]) extends LinearRegressorModel[K] {
  // https://en.wikipedia.org/wiki/Cochrane%E2%80%93Orcutt_estimation

  var betas: Array[MahoutVector] = _

  def predict(drmPredictors: DrmLike[K]): DrmLike[K] = {
    regressor.predict(drmPredictors)
  }

}

class CochraneOrcutt[K](hyperparameters: (Symbol, Any)*)  extends LinearRegressorModelFactory[K] {

  var regressor: LinearRegressorModelFactory[K] = hyperparameters.asInstanceOf[Map[Symbol,
    LinearRegressorModelFactory[K]]].getOrElse('regressor, new OrdinaryLeastSquares[K]())
  var iterations: Int = hyperparameters.asInstanceOf[Map[Symbol, Int]].getOrElse('iterations, 3)
  var cacheHint: CacheHint.CacheHint = hyperparameters.asInstanceOf[Map[Symbol, CacheHint.CacheHint]].getOrElse('cacheHint, CacheHint.MEMORY_ONLY)
  // For larger inputs, CacheHint.MEMORY_AND_DISK2 is reccomended.

  def setHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    setStandardHyperparameters(hyperparameters.toMap)
    regressor = hyperparameters.asInstanceOf[Map[Symbol, LinearRegressorModelFactory[K]]].getOrElse('regressor, new OrdinaryLeastSquares())
    regressor.calcStandardErrors = false
    regressor.calcCommonStatistics = false
    iterations = hyperparameters.asInstanceOf[Map[Symbol, Int]].getOrElse('iterations, 3)
    cacheHint = hyperparameters.asInstanceOf[Map[Symbol, CacheHint.CacheHint]].getOrElse('cacheHint, CacheHint.MEMORY_ONLY)
  }

  setHyperparameters(hyperparameters.toMap)

  def fit(drmFeatures: DrmLike[K], drmTarget: DrmLike[K], hyperparameters: (Symbol, Any)*): CochraneOrcuttModel[K] = {

    var hyperparameters: Option[Map[String,Any]] = None
    val betas = new Array[MahoutVector](iterations)
    var regressionModel: LinearRegressorModel[K] = regressor.fit(drmFeatures, drmTarget)
    betas(0) = regressionModel.beta

    val drmY = drmTarget
    val n = safeToNonNegInt(drmTarget.nrow)
    val Y = drmTarget(1 until n, 0 until 1).checkpoint(cacheHint)
    val Y_lag = drmTarget(0 until n - 1, 0 until 1).checkpoint(cacheHint)
    val X = drmFeatures(1 until n, 0 until 1).checkpoint(cacheHint)
    val X_lag = drmFeatures(0 until n - 1, 0 until 1).checkpoint(cacheHint)
    for (i <- 1 until iterations){
      val error = drmTarget - regressionModel.predict(drmFeatures)
      regressionModel = regressor.fit(drmFeatures, drmTarget)
      val rho = regressionModel.beta.get(0)

      val drmYprime = Y - Y_lag * rho
      val drmXprime = X - X_lag * rho

      if (i == iterations - 1 ){
        regressor.calcStandardErrors = true
        regressor.calcCommonStatistics = true
      }
      regressionModel = regressor.fit(drmFeatures, drmTarget)
      var betaPrime = regressionModel.beta
      val b0 = betaPrime(0) / (1 - rho)
      betaPrime(0) = b0
      betas(i) = betaPrime
    }

    val model = new CochraneOrcuttModel[K](regressionModel)
    model.betas = betas
    model.summary = (0 until iterations).map(i ⇒ s"Beta estimates on iteration " + i + ": "
      + model.betas.toString + "\n").mkString("") + "\n\n" + "Final Model:\n\n" + regressionModel.summary

    model
  }

}