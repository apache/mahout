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

package org.apache.mahout.math.algorithms.classifier

import org.apache.mahout.math.Vector
import org.apache.mahout.math.algorithms.neuralnet.mlp.DistributedMLPFitter
import org.apache.mahout.math.algorithms.preprocessing.{AsFactor, AsFactorModel}
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.scalabindings.dvec

import org.apache.mahout.math.algorithms.neuralnet.mlp.{DistributedMLPModel, DistributedMLPFitter, InCoreMLP}
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm._
import org.apache.mahout.math._
import org.apache.mahout.math.algorithms.neuralnet.Converters
import org.apache.mahout.math.drm._
import org.apache.mahout.math.{Vector => MahoutVector}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import MahoutCollections._

class DistributedMLPClassifier[K] extends ClassifierFitter[K] {

  var hiddenArch: Array[Int] = _
  var microIters: Int = _
  var macroIters: Int = _
  var factorizerModel: AsFactorModel = _
  var useBiases: Boolean = _

  def setStandardHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    hiddenArch = hyperparameters.asInstanceOf[Map[Symbol, Array[Int]]].getOrElse('hiddenArchitecture, Array(10))
    microIters = hyperparameters.asInstanceOf[Map[Symbol, Int]].getOrElse('microIters, 10)
    macroIters = hyperparameters.asInstanceOf[Map[Symbol, Int]].getOrElse('macroIters, 10)
    useBiases =  hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('useBiases, true)
  }

  def fit(drmX: DrmLike[K],
          drmTarget: DrmLike[K],
          hyperparameters: (Symbol, Any)*): DistributedMLPClassifierModel[K] = {


    factorizerModel = new AsFactor().fit(drmTarget)
    val factoredDrm = factorizerModel.transform(drmTarget)
    val dataDrm = drmX cbind factoredDrm

    val arch: Vector = dvec(Array(drmX.ncol.toDouble) ++ hiddenArch.map(i => i.toDouble) ++ Array(factoredDrm.ncol.toDouble))
    val distributedMLP = new DistributedMLPFitter[K](arch = arch,
      microIters = microIters,
      macroIters = macroIters,
      offsets = dvec(0, drmX.ncol, drmX.ncol, factoredDrm.ncol),
      useBiases)

    distributedMLP.fit(dataDrm)
    new DistributedMLPClassifierModel[K](distributedMLP.createDistributedModel(), factorizerModel)
  }
}

class DistributedMLPClassifierModel[K](mlpModel: DistributedMLPModel[K],
                                       factorizerModel: AsFactorModel) extends ClassifierModel[K] {

  /**
    * The raw predictions of the output neurons
    * @param drmPredictors
    * @return
    */
  def rawPredictions(drmPredictors: DrmLike[K]): DrmLike[K] = {
    mlpModel.predict(drmPredictors)
  }

  def predict(drmPredictors: DrmLike[K]): DrmLike[K] = {
    val rawScores = rawPredictions(drmPredictors)
    // oneHots = somefunc(rawScores) todo somefunc which converts dvec(0.001, 0.004, 2.41) to dvec(0,0,1)
    // factorizerModel.invTransform(oneHots)
    rawScores
  }

  def exportIncoreModel(): InCoreMLP = {
    mlpModel.exportIncoreModel()
  }

}