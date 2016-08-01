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


class OLS extends Regressor{
  // https://en.wikipedia.org/wiki/Ordinary_least_squares
  def fit[Int](drmY: DrmLike[Int], drmX: DrmLike[Int]) = {

    if (drmX.nrow != drmY.nrow){
      "throw an error here"
    }

    val drmXtX = drmX.t %*% drmX

    val drmXty = drmX.t %*% drmY

    // maybe some sort of flag to force this in core, would need an optimizer to determine if
    // its a good idea
    // val XtX = drmXtX.collect
    // val Xty = drmXty.collect(::, 0)

    fitParams("beta") = solve(drmXtX, drmXty)(::, 0)
    // add standard errors
    isFit = true
  }

  def predict[Int](drmX: DrmLike[Int]): DrmLike[Int] = {
    // throw warning if not fit
    drmX %*% fitParams.get("beta").get
  }

  def summary() = {
    "pass"
  }
}
