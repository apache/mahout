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
import org.apache.mahout.math.drm.CacheHint
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.RLikeOps._

class CochraneOrcutt[K](hyperparameters: Map[String, Any] = Map("" -> None)) extends LinearRegressor[K] {
  // https://en.wikipedia.org/wiki/Cochrane%E2%80%93Orcutt_estimation

  var regressor: LinearRegressor[K] = hyperparameters.asInstanceOf[Map[String, LinearRegressor[K]]].getOrElse("regressor", new OrdinaryLeastSquares())
  var iterations: Int = hyperparameters.asInstanceOf[Map[String, Int]].getOrElse("iterations", 3)
  var cacheHint: CacheHint.CacheHint = hyperparameters.asInstanceOf[Map[String, CacheHint.CacheHint]].getOrElse("cacheHint", CacheHint.MEMORY_ONLY)
  // For larger inputs, CacheHint.MEMORY_AND_DISK2 is reccomended.

  var betas: Array[MahoutVector] = _

  var summary = ""

  def fit(drmFeatures: DrmLike[K],
          drmTarget: DrmLike[K],
          hyperparameters: Map[String, Any] = Map("" -> None)): Unit = {

    var hyperparameters: Option[Map[String,Any]] = None
    betas = new Array[MahoutVector](iterations)
    regressor.fit(drmTarget, drmFeatures)
    betas(0) = regressor.beta

    drmY = drmTarget

    val Y = drmTarget(1 until drmTarget.nrow.toInt, 0 until 1).checkpoint(cacheHint)
    val Y_lag = drmTarget(0 until drmTarget.nrow.toInt - 1, 0 until 1).checkpoint(cacheHint)
    val X = drmFeatures(1 until drmFeatures.nrow.toInt, 0 until 1).checkpoint(cacheHint)
    val X_lag = drmFeatures(0 until drmFeatures.nrow.toInt - 1, 0 until 1).checkpoint(cacheHint)
    for (i <- 1 until iterations){
      val error = drmTarget - regressor.predict(drmFeatures)
      regressor.fit(error(1 until error.nrow.toInt, 0 until 1), error(0 until error.nrow.toInt - 1, 0 until 1))
      val rho = regressor.beta.get(0)

      val drmYprime = Y - Y_lag * rho
      val drmXprime = X - X_lag * rho

      regressor.fit(drmYprime, drmXprime)
      var betaPrime = regressor.beta
      val b0 = betaPrime(0) / (1 - rho)
      betaPrime(0) = b0
      betas(i) = betaPrime
    }

    summary = (0 until iterations).map(i => s"Beta estimates on iteration " + i + ": "
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
