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

import org.apache.mahout.math.algorithms.Model
import org.apache.mahout.math.algorithms.transformer.MeanCenter
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.function.Functions.SQUARE
import org.apache.mahout.math.scalabindings.RLikeOps._
import scala.reflect.ClassTag

class CoefficientOfDetermination extends Model {
  // https://en.wikipedia.org/wiki/Coefficient_of_determination

  var r2 = -1.0

  def fit[K: ClassTag](residuals: DrmLike[K], actuals: DrmLike[K]): Unit = {
    val sumSquareResiduals = residuals.assign(SQUARE).sum
    val mc = new MeanCenter()
    mc.fit(actuals)
    val totalResiduals = mc.transform(actuals)
    val sumSquareTotal = totalResiduals.assign(SQUARE).sum
    r2 = 1 - (sumSquareResiduals / sumSquareTotal)
  }

  def summary() = {
    // throw error if isFit = False
    "Coefficient of Determination (R-squared): " + r2
  }

}
