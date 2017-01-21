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
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.reflect.ClassTag

/**
  * import org.apache.mahout.math.algorithms.regression.OrdinaryLeastSquares
  * val model = new OrdinaryLeastSquares()
  *
  * model.calcStandardErrors = true
  * 
  */
class OrdinaryLeastSquares[K](hyperparameters: Map[String, Any] = Map("" -> None)) extends LinearRegressor[K] {
  // https://en.wikipedia.org/wiki/Ordinary_least_squares

  var calcStandardErrors: Boolean = hyperparameters.asInstanceOf[Map[String, Boolean]].getOrElse("calcStandardErrors", true)
  var addIntercept: Boolean = hyperparameters.asInstanceOf[Map[String, Boolean]].getOrElse("addIntercept", true)

  var summary = ""
  def fit(drmFeatures: DrmLike[K], drmTarget: DrmLike[K]): Unit = {

    if (drmFeatures.nrow != drmTarget.nrow){
      "throw an error here"
    }

    var X = drmFeatures
    if (addIntercept) {
      X = X cbind 1
    }
    val drmXtXinv = solve(X.t %*% X)

    val drmXty = (X.t %*% drmTarget).collect // this fails when number of columns^2 size matrix won't fit in driver

    beta = (drmXtXinv %*% drmXty)(::, 0)
    val k = X.ncol

    if (calcStandardErrors) {
      import org.apache.mahout.math.function.Functions.SQRT
      import org.apache.mahout.math.scalabindings.MahoutCollections._

      residuals = drmTarget - (X %*% beta)
      val ete = (residuals.t %*% residuals).collect // 1x1
      val n = drmTarget.nrow

      val invDegFreedomKindOf = (1.0 / (n - k))
      val varCovarMatrix = invDegFreedomKindOf * ete(0,0) * drmXtXinv
      se = varCovarMatrix.viewDiagonal.assign(SQRT)
      tScore = beta / se
      val tDist = new org.apache.commons.math3.distribution.TDistribution((n-k))
      pval = dvec(tScore.toArray.map(t => 2 * (1.0 - tDist.cumulativeProbability(t)) ))
      degreesFreedom = k
      summary = "Coef.\t\tEstimate\t\tStd. Error\t\tt-score\t\t\tPr(Beta=0)\n" +
        (0 until k).map(i => s"X${i}\t${beta(i)}\t${se(i)}\t${tScore(i)}\t${pval(i)}").mkString("\n")
    } else {
      summary = "Coef.\t\tEstimate\n" +
        (0 until k).map(i => s"X${i}\t${beta(i)}").mkString("\n")
    }

    if (addIntercept) {
      summary = summary.replace(s"X${k-1}", "(Intercept)")
    }

    isFit = true
  }

  def predict(drmPredictors: DrmLike[K]): DrmLike[K] = {
    // throw warning if not fit
    drmPredictors %*% beta
  }


}
