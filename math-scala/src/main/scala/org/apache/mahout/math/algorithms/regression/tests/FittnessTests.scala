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

import org.apache.mahout.math.algorithms.regression.RegressorModel
import org.apache.mahout.math.algorithms.preprocessing.MeanCenter
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.function.Functions.SQUARE
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.language.higherKinds
import scala.reflect.ClassTag

object FittnessTests {

  // https://en.wikipedia.org/wiki/Coefficient_of_determination
  def CoefficientOfDetermination[R[K] <: RegressorModel[K], K](model: R[K],
                                                               drmTarget: DrmLike[K]): R[K] = {
    val sumSquareResiduals = model.residuals.assign(SQUARE).sum
    val mc = new MeanCenter()
    val totalResiduals = mc.fitTransform(drmTarget)
    val sumSquareTotal = totalResiduals.assign(SQUARE).sum
    val r2 = 1 - (sumSquareResiduals / sumSquareTotal)
    model.testResults += ('r2 -> r2)  // need setResult and setSummary method incase you change in future, also to initialize map if non exists or update value if it does

    model.summary += s"\nR^2: ${r2}"
    model
  }

  // https://en.wikipedia.org/wiki/Mean_squared_error
  def MeanSquareError[R[K] <: RegressorModel[K], K](model: R[K]): R[K] = {
    val mse = model.residuals.assign(SQUARE).sum / model.residuals.nrow
    model.testResults += ('mse -> mse)
    model.summary += s"\nMean Squared Error: ${mse}"
    model
  }
}
