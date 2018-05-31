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

import org.apache.mahout.math.algorithms.regression.tests._
import org.apache.mahout.math.drm.{CacheHint, DrmLike, safeToNonNegInt}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.function.Functions
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.{Vector => MahoutVector}


class CochraneOrcuttModel[K](regressor: LinearRegressorModel[K]) extends LinearRegressorModel[K] {
  // https://en.wikipedia.org/wiki/Cochrane%E2%80%93Orcutt_estimation

  var betas: Array[MahoutVector] = _
  var dws: Array[Double] = _
  var rhos: Array[Double] = _

  def predict(drmPredictors: DrmLike[K]): DrmLike[K] = {
    regressor.predict(drmPredictors)
  }

}

class CochraneOrcutt[K](hyperparameters: (Symbol, Any)*)  extends LinearRegressorFitter[K] {

  var regressor: LinearRegressorFitter[K] = _
  var iterations: Int = _
  var cacheHint: CacheHint.CacheHint = _
  // For larger inputs, CacheHint.MEMORY_AND_DISK2 is reccomended.

  def setHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    setStandardHyperparameters(hyperparameters.toMap)
    regressor = hyperparameters.asInstanceOf[Map[Symbol, LinearRegressorFitter[K]]].getOrElse('regressor, new OrdinaryLeastSquares())
    regressor.calcStandardErrors = false
    regressor.calcCommonStatistics = false
    iterations = hyperparameters.asInstanceOf[Map[Symbol, Int]].getOrElse('iterations, 3)
    cacheHint = hyperparameters.asInstanceOf[Map[Symbol, CacheHint.CacheHint]].getOrElse('cacheHint, CacheHint.MEMORY_ONLY)
  }

  setHyperparameters(hyperparameters.toMap)

  def calculateRho(errorDrm: DrmLike[K]): Double ={
    val error = errorDrm.collect.viewColumn(0)
    val n = error.length - 1
    val e2: MahoutVector = error.viewPart(1, n)
    val e3: MahoutVector = error.viewPart(0, n)
    // regression through the origin lm(e2 ~e3 -1) is sum(e2 * e3) / e3^2
    e3.times(e2).sum / e3.assign(Functions.SQUARE).sum
  }

  def fit(drmFeatures: DrmLike[K], drmTarget: DrmLike[K], hyperparameters: (Symbol, Any)*): CochraneOrcuttModel[K] = {

    setHyperparameters(hyperparameters.toMap[Symbol, Any])

    val betas = new Array[MahoutVector](iterations)
    val models = new Array[LinearRegressorModel[K]](iterations)
    val dws = new Array[Double](iterations)
    val rhos = new Array[Double](iterations)

    val n = safeToNonNegInt(drmTarget.nrow)
    val Y = drmTarget(1 until n, 0 until 1).checkpoint(cacheHint)
    val Y_lag = drmTarget(0 until n - 1, 0 until 1).checkpoint(cacheHint)
    val X = drmFeatures(1 until n, 0 until drmFeatures.ncol).checkpoint(cacheHint)
    val X_lag = drmFeatures(0 until n - 1, 0 until drmFeatures.ncol).checkpoint(cacheHint)

    // Step 1: Normal Regression
    regressor.calcStandardErrors = true
    regressor.calcCommonStatistics = true
    models(0) = regressor.fit(drmFeatures, drmTarget)
    regressor.calcStandardErrors = false
    regressor.calcCommonStatistics = false
    betas(0) = models(0).beta
    var residuals = drmTarget - models(0).predict(drmFeatures)

    for (i <- 1 until iterations){
      // Step 2: Calculate Rho
      val rho_hat = calculateRho(residuals)
      rhos(i-1) = rho_hat

      // Step 3: Transform Variables
      val drmYprime = Y - (Y_lag * rho_hat)
      val drmXprime = X - (X_lag * rho_hat)

      // Step 4: Get Estimates of Transformed Equation
      if (i == iterations - 1 ){
        // get standard errors on last iteration only
        regressor.calcStandardErrors = true
        regressor.calcCommonStatistics = true
      }
      models(i) = regressor.fit(drmXprime, drmYprime)
      // Make this optional- only for parity with R reported dw-stat, doesn't really mean anything
      dws(i) = AutocorrelationTests.DurbinWatson( models(i),
                                                  drmTarget - models(i).predict(drmFeatures))
        .testResults.get('durbinWatsonTestStatistic).get.asInstanceOf[Double]

      models(i).beta(X.ncol) = models(i).beta(X.ncol) / (1 - rho_hat) // intercept adjust
      betas(i) = models(i).beta

      // Step 5: Use Betas from (4) to recalculate model from (1)
      residuals = drmTarget - models(i).predict(drmFeatures)

      /** Step 6: repeat  Step 2 through 5 until a stopping criteria is met.
        *  some models call for convergence-
        *   Kunter et. al reccomend 3 iterations, if you don't achieve desired results, use
        *   an alternative method.
        **/
    }

    var finalModel = new CochraneOrcuttModel[K](models(iterations -1))
    finalModel.betas = betas
    finalModel.dws = dws
    finalModel.rhos = rhos
    finalModel.tScore = models(iterations -1).tScore
    finalModel.pval =  models(iterations -1).pval
    finalModel.beta = models(iterations -1).beta
    val se = models(iterations -1).se
    se(se.length -1) = se(se.length -1) / (1 - rhos(iterations - 2))
    finalModel.se = se
    finalModel.summary = "Original Model:\n" + models(0).summary +
    "\n\nTransformed Model:\n" +
      generateSummaryString(finalModel) +
    "\n\nfinal rho: " + finalModel.rhos(iterations - 2) +
    s"\nMSE: ${models(iterations -1 ).mse}\nR2: ${models(iterations -1 ).r2}\n"

    if (models(0).addIntercept == true){
      finalModel.summary = finalModel.summary.replace(s"X${X.ncol}", "(Intercept)")
    }

    finalModel
  }

}