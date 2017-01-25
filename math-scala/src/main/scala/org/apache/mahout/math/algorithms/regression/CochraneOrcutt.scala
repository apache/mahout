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

class CochraneOrcutt[K](hyperparameters: (Symbol, Any)*) extends LinearRegressor[K] {
  // https://en.wikipedia.org/wiki/Cochrane%E2%80%93Orcutt_estimation

  var regressor: LinearRegressor[K] = hyperparameters.asInstanceOf[Map[Symbol, LinearRegressor[K]]].getOrElse('regressor, new OrdinaryLeastSquares())
  var iterations: Int = hyperparameters.asInstanceOf[Map[Symbol, Int]].getOrElse('iterations, 3)
  var cacheHint: CacheHint.CacheHint = hyperparameters.asInstanceOf[Map[Symbol, CacheHint.CacheHint]].getOrElse('cacheHint, CacheHint.MEMORY_ONLY)
  // For larger inputs, CacheHint.MEMORY_AND_DISK2 is reccomended.

  var betas: Array[MahoutVector] = _

  var summary = ""

  setHyperparameters(hyperparameters.toMap)

  def setHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    regressor = hyperparameters.asInstanceOf[Map[Symbol, LinearRegressor[K]]].getOrElse('regressor, new OrdinaryLeastSquares())
    iterations = hyperparameters.asInstanceOf[Map[Symbol, Int]].getOrElse('iterations, 3)
    cacheHint = hyperparameters.asInstanceOf[Map[Symbol, CacheHint.CacheHint]].getOrElse('cacheHint, CacheHint.MEMORY_ONLY)
  }

  def fit(drmFeatures: DrmLike[K], drmTarget: DrmLike[K], hyperparameters: (Symbol, Any)*): Unit = {

    var hyperparameters: Option[Map[String,Any]] = None
    betas = new Array[MahoutVector](iterations)
    regressor.fit(drmFeatures, drmTarget)
    betas(0) = regressor.beta

    drmY = drmTarget
    val n = safeToNonNegInt(drmTarget.nrow)
    val Y = drmTarget(1 until n, 0 until 1).checkpoint(cacheHint)
    val Y_lag = drmTarget(0 until n - 1, 0 until 1).checkpoint(cacheHint)
    val X = drmFeatures(1 until n, 0 until 1).checkpoint(cacheHint)
    val X_lag = drmFeatures(0 until n - 1, 0 until 1).checkpoint(cacheHint)
    for (i <- 1 until iterations){
      val error = drmTarget - regressor.predict(drmFeatures)
      regressor.fit(drmFeatures, drmTarget)
      val rho = regressor.beta.get(0)

      val drmYprime = Y - Y_lag * rho
      val drmXprime = X - X_lag * rho

      regressor.fit(drmFeatures, drmTarget)
      var betaPrime = regressor.beta
      val b0 = betaPrime(0) / (1 - rho)
      betaPrime(0) = b0
      betas(i) = betaPrime
    }

    summary = (0 until iterations).map(i ⇒ s"Beta estimates on iteration " + i + ": "
      + betas.toString + "\n").mkString("") + "\n\n" + "Final Model:\n\n" + regressor.summary

    isFit = true
  }

  def predict(drmPredictors: DrmLike[K]): DrmLike[K] = {
    if (!isFit){
      throw new Exception("Model hasn't been fit yet- please run .fit(...) method first.")
    }
    regressor.predict(drmPredictors)
  }

}
