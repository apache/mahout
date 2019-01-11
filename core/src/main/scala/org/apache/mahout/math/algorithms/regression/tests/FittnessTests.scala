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

package org.apache.mahout.math.algorithms.regression.tests

import org.apache.commons.math3.distribution.FDistribution
import org.apache.mahout.math.algorithms.regression.RegressorModel
import org.apache.mahout.math.algorithms.preprocessing.MeanCenter
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.function.Functions.SQUARE
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._

import scala.language.higherKinds

object FittnessTests {

  // https://en.wikipedia.org/wiki/Coefficient_of_determination
  def CoefficientOfDetermination[R[K] <: RegressorModel[K], K](model: R[K],
                                                               drmTarget: DrmLike[K],
                                                               residuals: DrmLike[K]): R[K] = {
    val sumSquareResiduals = residuals.assign(SQUARE).sum
    val mc = new MeanCenter()
    val totalResiduals = mc.fitTransform(drmTarget)
    val sumSquareTotal = totalResiduals.assign(SQUARE).sum
    val r2 = 1 - (sumSquareResiduals / sumSquareTotal)
    model.r2 = r2
    model.testResults += ('r2 -> r2)  // need setResult and setSummary method incase you change in future, also to initialize map if non exists or update value if it does
    //model.summary += s"\nR^2: ${r2}"
    model
  }

  // https://en.wikipedia.org/wiki/Mean_squared_error
  def MeanSquareError[R[K] <: RegressorModel[K], K](model: R[K], residuals: DrmLike[K]): R[K] = {
    // TODO : I think mse denom should be (row - col) ?? <-- https://en.wikipedia.org/wiki/Mean_squared_error  see regression section
    val mse = residuals.assign(SQUARE).sum / residuals.nrow
    model.mse = mse
    model.testResults += ('mse -> mse)
    //model.summary += s"\nMean Squared Error: ${mse}"
    model
  }

  // Since rss is needed for multiple test statistics, use this function to cache this value
  def calculateResidualSumOfSquares[R[K] <: RegressorModel[K], K](model: R[K],residuals: DrmLike[K]) : R[K] = {
    // This is a check so that model.rss isnt unnecessarily computed
    // by default setting this value to negative, so that the first time its garaunteed to evaluate.
    if (model.rss < 0) {
      val ete = (residuals.t %*% residuals).collect // 1x1
      model.rss = ete(0, 0)
    }
    model
  }


  // https://en.wikipedia.org/wiki/F-test
  /*
  # R Prototype
  # Cereal Dataframe
  df1 <- data.frame(
    "X0" = c(1,1,1,1,1,1,1,1,1),
    "a"  = c(2,1,1,2,1,2,6,3,3),
    "b" = c( 2,2,1,1,2,1,2,2,3),
    "c" = c( 10.5,12,12, 11,12, 16,17, 13,13),
    "d" = c( 10,12,13,13,11,8, 1, 7, 4),
    "target" = c( 29.509541,18.042851,22.736446,32.207582,21.871292,36.187559,50.764999,40.400208,45.811716))

  # Create linear regression models adding features one by one
  lrfit0 <- lm(data=df1, formula = target ~ 1  )
  lrfit1 <- lm(data=df1, formula = target ~ a  )
  lrfit2 <- lm(data=df1, formula = target ~ a + b )
  lrfit3 <- lm(data=df1, formula = target ~ a + b + c )
  lrfit4 <- lm(data=df1, formula = target ~ a + b + c + d)

  ######################################
  # Fscore Calculation
  ######################################

  # So in the anova report using lm ...
  # These are the residual sum of squares for each model
  rssint <- sum(lrfit0$residuals^2)
  rssa <- sum(lrfit1$residuals^2)
  rssb <- sum(lrfit2$residuals^2)
  rssc <- sum(lrfit3$residuals^2)
  rssd <- sum(lrfit4$residuals^2)

  #Ftest in overall model
  (rssint - rssd)/4 / (rssd/4)  # g = 4, n - g - 1  = 4
  # Compare with R
  summary(lrfit4)

   */
  def FTest[R[K] <: RegressorModel[K], K](model: R[K] , drmTarget: DrmLike[K]): R[K] = {

    val targetMean: Double = drmTarget.colMeans().get(0)

    // rssint is the Residual Sum of Squares for model using only based on the intercept
    val rssint: Double = ((drmTarget - targetMean  ).t %*% (drmTarget - targetMean)).zSum()
    // K-1 is model.degreesOfFreedom-1
    // N-K is model.trainingExamples - model.degreesOfFreedom

    val fScore = ((rssint - model.rss) / (model.degreesOfFreedom-1) / ( model.rss / (model.trainingExamples - model.degreesOfFreedom)))
    val fDist = new FDistribution(model.degreesOfFreedom-1,model.trainingExamples-model.degreesOfFreedom)
    val fpval = 1.0 - fDist.cumulativeProbability(fScore)
    model.fpval = fpval

    model.fScore = fScore
    model.testResults += ('fScore -> fScore)
    //model.summary += s"\nFscore : ${fScore}"
    model
  }


}
