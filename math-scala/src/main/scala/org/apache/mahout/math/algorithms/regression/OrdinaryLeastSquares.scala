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

/**
  * import org.apache.mahout.math.algorithms.regression.OrdinaryLeastSquares
  * val model = new OrdinaryLeastSquares()
  *
  * model.calcStandardErrors = true
  * 
  */
class OrdinaryLeastSquares extends Regressor{
  // https://en.wikipedia.org/wiki/Ordinary_least_squares

  var calcStandardErrors = true

  var addIntercept = true

  def fit[Int](drmY: DrmLike[Int], drmX: DrmLike[Int]) = {

    if (drmX.nrow != drmY.nrow){
      "throw an error here"
    }

    var X = drmX
    if (addIntercept) {
      X = X cbind 1
    }
    val drmXtXinv = solve(X.t %*% X)

    val drmXty = (X.t %*% drmY).collect // this fails when number of columns^2 size matrix won't fit in driver

    val beta = (drmXtXinv %*% drmXty)(::, 0)

    if (calcStandardErrors) {
      import org.apache.mahout.math.function.Functions.SQRT
      import org.apache.mahout.math.scalabindings.MahoutCollections._

      val e = (drmY - X %*% beta).collect
      val ete = e.t %*% e
      val n = drmY.nrow
      val k = X.ncol
      val invDegFreedomKindOf = (1.0 / (n - k))
      val varCovarMatrix = invDegFreedomKindOf * ete(0,0) * drmXtXinv
      val se = varCovarMatrix.viewDiagonal.assign(SQRT)
      val tScore = beta / se
      val tDist = new org.apache.commons.math3.distribution.TDistribution((n-k))
      val pval = dvec(tScore.toArray.map(t => 2 * (1.0 - tDist.cumulativeProbability(t)) ))
      fitParams("se") = se
      fitParams("tScore") = tScore
      fitParams("pval") = pval
      fitParams("degreesFreedom") = dvec(k)
    }
    fitParams("beta") = beta
    isFit = true
  }

  def predict[Int](drmX: DrmLike[Int]): DrmLike[Int] = {
    // throw warning if not fit
    drmX %*% fitParams.get("beta").get
  }

  def summary() = {
    val beta = fitParams("beta")
    val se = fitParams("se")
    val tScore = fitParams("tScore")
    val pval = fitParams("pval")
    val k = fitParams("degreesFreedom").get(0).toInt

    var summaryString = "Coef.\t\tEstimate\t\tStd. Error\t\tt-score\t\t\tPr(Beta=0)\n" +
      (0 until k).map(i => s"X${i}\t${beta(i)}\t${se(i)}\t${tScore(i)}\t${pval(i)}").mkString("\n")

    if (addIntercept) {
      summaryString = summaryString.replace(s"X${k-1}", "(Intercept)")
    }

    summaryString
  }
}
