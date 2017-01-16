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

import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.RLikeOps._

class CochraneOrcutt extends Regressor {
  // https://en.wikipedia.org/wiki/Cochrane%E2%80%93Orcutt_estimation

  var regressor : Regressor = new OrdinaryLeastSquares() // type of regression to do- must have a 'beta' fit param
  var iterations = 3 // Number of iterations to run

  def fit[Int](drmY: DrmLike[Int], drmX: DrmLike[Int]) = {

    regressor.fit(drmY, drmX)
    fitParams("beta0") = regressor.fitParams("beta")

    val Y = drmY(1 until drmY.nrow.toInt, 0 until 1).checkpoint()
    val Y_lag = drmY(0 until drmY.nrow.toInt - 1, 0 until 1).checkpoint()
    val X = drmX(1 until drmX.nrow.toInt, 0 until 1).checkpoint()
    val X_lag = drmX(0 until drmX.nrow.toInt - 1, 0 until 1).checkpoint()
    for (i <- 1 to iterations){
      val error = drmY - regressor.predict(drmX)
      regressor.fit(error(1 until error.nrow.toInt, 0 until 1),
                    error(0 until error.nrow.toInt - 1, 0 until 1))
      val rho = regressor.fitParams("beta").get(0)

      val drmYprime = Y - Y_lag * rho
      val drmXprime = X - X_lag * rho

      regressor.fit(drmYprime, drmXprime)
      var betaPrime = regressor.fitParams("beta")
      val b0 = betaPrime(0) / (1 - rho)
      betaPrime(0) = b0
      fitParams("beta_" + i) = betaPrime
    }

    isFit = true
  }

  def predict[Int](drmX: DrmLike[Int]): DrmLike[Int] = {
    regressor.predict(drmX)
  }

  def summary() = {
    "pass"
  }

}
