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
    model.summary += s"\nR^2: ${r2}"
    model
  }

  // https://en.wikipedia.org/wiki/Mean_squared_error
  def MeanSquareError[R[K] <: RegressorModel[K], K](model: R[K], residuals: DrmLike[K]): R[K] = {
    // TODO : I think mse denom should be (row - col) ??
    val mse = residuals.assign(SQUARE).sum / residuals.nrow
    model.mse = mse
    model.testResults += ('mse -> mse)
    model.summary += s"\nMean Squared Error: ${mse}"
    model
  }

  // https://en.wikipedia.org/wiki/xxxx
  def FTest[R[K] <: RegressorModel[K], K](model: R[K],  drmFeatures: DrmLike[K], drmTarget: DrmLike[K]): R[K] = {

    // This is the residual sum of squares for just the intercept
    //println(" drmTarget.ncol) = " +  drmTarget.ncol)
    val interceptCol = drmTarget.ncol - 1
    //val targetMean: Double = drmTarget
    val targetMean: Double = drmTarget.colMeans().get(0)

    val rssint: Double = ((drmTarget - targetMean  ).t %*% (drmTarget - targetMean)).zSum()
    // ete above is the RSS for the calculated model

    //println(" model.beta(0) = " +  model.beta(0))
    //println(" model.beta(interceptCol) = " +  model.beta(interceptCol))
    //println("rssint = " + rssint)
    //println("rssmod = " + rssmod)

    val groupDof = drmFeatures.ncol-1
    val fScore = ((rssint - model.rss) / groupDof) / ( model.rss / (drmFeatures.nrow - groupDof- 1 ))
    //println("groupDof = " + groupDof)
    //println("fScore = " + fScore)
    val fDist = new FDistribution(groupDof,drmTarget.nrow-groupDof-1)
    val fpval = 1.0 - fDist.cumulativeProbability(fScore)
    model.fpval = fpval

    model.fScore = fScore
    model.testResults += ('fScore -> fScore)
    model.summary += s"\nFscore : ${fScore}"
    model
  }


}
