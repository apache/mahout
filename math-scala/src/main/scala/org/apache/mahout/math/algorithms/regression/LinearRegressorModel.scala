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

import org.apache.mahout.math.algorithms.regression.tests.FittnessTests
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.dvec
import org.apache.mahout.math.{Matrix, Vector => MahoutVector}
import org.apache.mahout.math.scalabindings.RLikeOps._
import scala.language.higherKinds

trait LinearRegressorModel[K] extends RegressorModel[K] {

  var beta: MahoutVector = _
  var se: MahoutVector = _
  var tScore: MahoutVector = _
  var pval: MahoutVector = _
  var degreesFreedom: Int = _

}

trait LinearRegressorFitter[K] extends RegressorFitter[K] {

  var addIntercept: Boolean = _
  var calcStandardErrors: Boolean = _
  var calcCommonStatistics: Boolean = _

  def fit(drmX: DrmLike[K],
          drmTarget: DrmLike[K],
          hyperparameters: (Symbol, Any)*): LinearRegressorModel[K]


  def setStandardHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    calcCommonStatistics = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('calcCommonStatistics, true)
    calcStandardErrors = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('calcStandardErrors, true)
    addIntercept = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('addIntercept, true)
  }

  def calculateStandardError[M[K] <: LinearRegressorModel[K]](X: DrmLike[K],
                             drmTarget: DrmLike[K],
                             drmXtXinv: Matrix,
                             model: M[K]): M[K] = {
    import org.apache.mahout.math.function.Functions.SQRT
    import org.apache.mahout.math.scalabindings.MahoutCollections._
    var modelOut = model
    val yhat = X %*% model.beta
    val residuals = drmTarget - yhat
    val ete = (residuals.t %*% residuals).collect // 1x1
    val n = drmTarget.nrow
    val k = safeToNonNegInt(X.ncol)
    val invDegFreedomKindOf = 1.0 / (n - k)
    val varCovarMatrix = invDegFreedomKindOf * ete(0,0) * drmXtXinv
    val se = varCovarMatrix.viewDiagonal.assign(SQRT)
    val tScore = model.beta / se
    val tDist = new org.apache.commons.math3.distribution.TDistribution(n-k)
    val pval = dvec(tScore.toArray.map(t => 2 * (1.0 - tDist.cumulativeProbability(t)) ))
    // ^^ TODO bug in this calculation- fix and add test
    //degreesFreedom = k
    modelOut.summary = "Coef.\t\tEstimate\t\tStd. Error\t\tt-score\t\t\tPr(Beta=0)\n" +
      (0 until k).map(i => s"X${i}\t${model.beta(i)}\t${se(i)}\t${tScore(i)}\t${pval(i)}").mkString("\n")

    modelOut.se = se
    modelOut.tScore = tScore
    modelOut.pval = pval
    modelOut.degreesFreedom = X.ncol

    if (calcCommonStatistics){
      modelOut = calculateCommonStatistics(modelOut, drmTarget, residuals)
    }
    modelOut
  }

  def calculateCommonStatistics[M[K] <: LinearRegressorModel[K]](model: M[K],
                                                                 drmTarget: DrmLike[K],
                                                                 residuals: DrmLike[K]): M[K] ={
    var modelOut = model
    modelOut = FittnessTests.CoefficientOfDetermination(model, drmTarget, residuals)
    modelOut = FittnessTests.MeanSquareError(model, residuals)
    modelOut
  }

  def modelPostprocessing[M[K] <: LinearRegressorModel[K]](model: M[K],
                     X: DrmLike[K],
                     drmTarget: DrmLike[K],
                     drmXtXinv: Matrix): M[K] = {
    var modelOut = model
    if (calcStandardErrors) {
      modelOut = calculateStandardError(X, drmTarget, drmXtXinv, model )
    } else {
      modelOut.summary = "Coef.\t\tEstimate\n" +
        (0 until X.ncol).map(i => s"X${i}\t${modelOut.beta(i)}").mkString("\n")
      if (calcCommonStatistics) { // we do this in calcStandard errors to avoid calculating residuals twice
        val residuals = drmTarget - (X %*% modelOut.beta)
        modelOut = calculateCommonStatistics(modelOut, drmTarget, residuals)
      }

      modelOut
    }

    if (addIntercept) {
      model.summary.replace(s"X${X.ncol - 1}", "(Intercept)")
    }
    model
  }
}
