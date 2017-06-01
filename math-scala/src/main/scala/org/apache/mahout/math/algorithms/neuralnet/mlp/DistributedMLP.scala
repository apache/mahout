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

package org.apache.mahout.math.algorithms.neuralnet.mlp

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

class DistributedMLPFitter[K](arch: Vector,
                        microIters: Int = 10,
                        macroIters: Int = 50,
                        offsets: Vector,
                        useBiases: Boolean) {

  var pv: Vector = createInitialWeightVector()
  var useBaisesInt = 0

  def createInitialWeightVector(): Vector = {
    var initNetwork = new InCoreMLP()
    initNetwork.useBiases = useBiases
    initNetwork.createWeightsArray(arch)
    initNetwork.parameterVector().cloned
  }

  def partialFit(drmData: DrmLike[K]): Unit = {
    implicit val ctx = drmData.context
    implicit val ktag =  drmData.keyClassTag

    val bcArch = drmBroadcast(arch)
    val bcMicroIters = drmBroadcast(dvec(microIters))
    val bcOffsets = drmBroadcast(offsets)

    if (useBiases){ useBaisesInt = 1 }

    val bcUseBiases = drmBroadcast( dvec( useBaisesInt ) )
    val bcWeightsArray = drmBroadcast(pv)

    val paramMatrix = drmData
      .allreduceBlock(
    { case (keys, block: Matrix) => {
      val inCoreNetwork = new InCoreMLP()
      val localOffsets = bcOffsets.value

      inCoreNetwork.inputStart = localOffsets.get(0).toInt
      inCoreNetwork.inputOffset = localOffsets.get(1).toInt
      inCoreNetwork.targetStart = localOffsets.get(2).toInt
      inCoreNetwork.targetOffset = localOffsets.get(3).toInt

      inCoreNetwork.useBiases = bcUseBiases.value.get(0) match {
        case 1 => true
        case 0 => false
      }

      inCoreNetwork.createWeightsArray(bcArch.value)
      inCoreNetwork.setParametersFromVector(bcWeightsArray.value)

      for (i <- 0 until bcMicroIters.value.get(0).toInt) {
        inCoreNetwork.forwardBackwardMatrix(block)
      }

      val v: Vector = dvec(Array(block.nrow.toDouble) ++ inCoreNetwork.parameterVector().toArray)
      val outputM = new DenseMatrix(block.nrow, v.length)

      outputM.assignRow(0, v)

      outputM

      }
      },
      { case (oldM: Matrix, newM: Matrix) => {
        val oldV = oldM.viewRow(0).viewPart(1, oldM.ncol -1)
        val newV = newM.viewRow(0).viewPart(1, newM.ncol -1)
        val total = oldM.get(0,0) + newM.get(0,0)
        val oldW = oldM.get(0,0) / total
        val newW = newM.get(0,0) / total
        val outputV = (oldV * oldW) + (newV * newW)

        dense(Array(total) ++ outputV.toArray)

    }})

    pv = paramMatrix.viewRow(0).viewPart(1, paramMatrix.ncol -1)

  }

  def fit(drmData: DrmLike[K]): Unit = {
    for (i <- 0 until macroIters) {
      partialFit(drmData)
    }
  }

  def createDistributedModel(): DistributedMLPModel[K] = {
    new DistributedMLPModel[K](arch, offsets, pv, useBaisesInt)
  }

}

class DistributedMLPModel[K](arch: Vector,
                             offsets: Vector,
                             pv: Vector,
                             useBiasesInt: Int) {



  def predict(drmData: DrmLike[K]): DrmLike[K] = {
    implicit val ctx = drmData.context

    val bcArch = drmBroadcast(arch)
    val bcOffsets = drmBroadcast(offsets)
    val bcWeightsArray = drmBroadcast(pv)

    val bcUseBiases = drmBroadcast( dvec( useBiasesInt ))

    implicit val ktag =  drmData.keyClassTag

    drmData.mapBlock(offsets.get(3).toInt) {
      case (keys, block: Matrix) => {
        val inCoreNetwork = new InCoreMLP()
        val localOffsets = bcOffsets.value

        inCoreNetwork.inputStart = localOffsets.get(0).toInt
        inCoreNetwork.inputOffset = localOffsets.get(1).toInt
        inCoreNetwork.targetStart = localOffsets.get(2).toInt
        inCoreNetwork.targetOffset = localOffsets.get(3).toInt

        inCoreNetwork.useBiases = bcUseBiases.value.get(0) match {
          case 1 => true
          case 0 => false
        }

        inCoreNetwork.createWeightsArray(bcArch.value)
        inCoreNetwork.setParametersFromVector(bcWeightsArray.value)

        val output = new DenseMatrix(block.nrow, inCoreNetwork.targetOffset)
        for (row <- 0 until block.nrow) {
          inCoreNetwork.feedForward(block(row, inCoreNetwork.inputStart until inCoreNetwork.inputOffset))
          output.assignRow(row, inCoreNetwork.A(inCoreNetwork.A.size -1))
        }
        (keys, output)
       }
    }
  }

  def exportIncoreModel(): InCoreMLP = {
    val inCoreNetwork = new InCoreMLP()

    inCoreNetwork.inputStart = offsets.get(0).toInt
    inCoreNetwork.inputOffset = offsets.get(1).toInt
    inCoreNetwork.targetStart = offsets.get(2).toInt
    inCoreNetwork.targetOffset = offsets.get(3).toInt

    inCoreNetwork.createWeightsArray(arch)
    inCoreNetwork.setParametersFromVector(pv)
    inCoreNetwork
  }

}