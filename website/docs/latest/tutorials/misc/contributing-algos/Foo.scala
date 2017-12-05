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


import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.function.VectorFunction
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.drm.DrmLike

class Foo[K] extends RegressorFitter[K] {

  var guessThisNumber: Double = _

  def setStandardHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    guessThisNumber = hyperparameters.asInstanceOf[Map[Symbol, Double]].getOrElse('guessThisNumber, 1.0)
  }
  def fit(drmX  : DrmLike[K],
          drmTarget: DrmLike[K],
          hyperparameters: (Symbol, Any)*): FooModel[K] ={
    /**
      * Normally one would have a lot more code here.
      */

    var model = new FooModel[K]

    setStandardHyperparameters(hyperparameters.toMap)
    model.guessThisNumber = guessThisNumber
    model.summary = s"This model will always guess ${model.guessThisNumber}"
    model
  }
}

class FooModel[K] extends RegressorModel[K] {

  var guessThisNumber: Double = _

  def predict(drmPredictors: DrmLike[K]): DrmLike[K] = {

    // This is needed for MapBlock
    implicit val ktag =  drmPredictors.keyClassTag
    // This is needed for broadcasting
    implicit val ctx = drmPredictors.context

    val bcGuess = drmBroadcast(dvec(guessThisNumber))
    drmPredictors.mapBlock(1) {
      case (keys, block: Matrix) => {
        var outputBlock = new DenseMatrix(block.nrow, 1)
        keys -> (outputBlock += bcGuess.value.get(0))
      }
    }
  }
}