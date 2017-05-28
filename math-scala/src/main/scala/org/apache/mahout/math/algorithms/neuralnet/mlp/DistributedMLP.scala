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

class DistributedMLP[K](arch: Vector,
                        microIters: Int = 10,
                        macroIters: Int = 50,
                        offsets: Vector) {

  var finalNetwork = new InCoreMLP()
  var pm: Matrix = _

  var microIterations = microIters

  finalNetwork.createWeightsArray(arch)
  var pv: Vector = finalNetwork.parameterVector().cloned

  def partialFit(drmData: DrmLike[K]) = {
    implicit val ctx = drmData.context
    implicit val ktag =  drmData.keyClassTag

    val bcArch = drmBroadcast(arch)
    val bcMicroIters = drmBroadcast(dvec(microIters))
    val bcOffsets = drmBroadcast(offsets)

    val bcU = drmBroadcast(pv)

    val outputSize = finalNetwork.parameterVector().length + 1
    val paramMatrix = drmData
      //.mapBlock(outputSize)
      .allreduceBlock(
    { case (keys, block: Matrix) => {
      val inCoreNetwork = new InCoreMLP()
      val localOffsets = bcOffsets.value

      inCoreNetwork.inputStart = localOffsets.get(0).toInt
      inCoreNetwork.inputOffset = localOffsets.get(1).toInt
      inCoreNetwork.targetStart = localOffsets.get(2).toInt
      inCoreNetwork.targetOffset = localOffsets.get(3).toInt

      inCoreNetwork.createWeightsArray(bcArch.value)
      inCoreNetwork.setParametersFromVector(bcU.value)

      for (i <- 0 until bcMicroIters.value.get(0).toInt) {
        inCoreNetwork.forwardBackwardMatrix(block)
      }

      val v: Vector = dvec(Array(block.nrow.toDouble) ++ inCoreNetwork.parameterVector().toArray)
      val outputM = new DenseMatrix(block.nrow, v.length)

      outputM.assignRow(0, v)

      //(keys, outputM)
      outputM
      //          block

      }
      },
      { case (oldM: Matrix, newM: Matrix) => {
        //          oldM
        val oldV = oldM.viewRow(0).viewPart(1, oldM.ncol -1)
        val newV = newM.viewRow(0).viewPart(1, newM.ncol -1)
        val total = oldM.get(0,0) + newM.get(0,0)
        val oldW = oldM.get(0,0) / total
        val newW = newM.get(0,0) / total
        val outputV = (oldV * oldW) + (newV * newW)

        dense(Array(total) ++ outputV.toArray)

    }})

    //pm = paramMatrix
    //finalNetwork.setParametersFromVector(paramMatrix.viewRow(0).viewPart(1, paramMatrix.ncol -1))
    //paramMatrix
    pv = paramMatrix.viewRow(0).viewPart(1, paramMatrix.ncol -1)

  }

  def reduceParts(drmData: DrmLike[K]) = {
    pv = drmData.allreduceBlock(
      { case (keys, block: Matrix) =>  block},
      { case (oldM: Matrix, newM: Matrix) => {
        //          oldM
        val oldV = oldM.viewRow(0).viewPart(1, oldM.ncol -1)
        val newV = newM.viewRow(0).viewPart(1, newM.ncol -1)
        val total = oldM.get(0,0) + newM.get(0,0)
        val oldW = oldM.get(0,0) / total
        val newW = newM.get(0,0) / total
        val outputV = (oldV * oldW) + (newV * newW)

        dense(Array(total) ++ outputV.toArray)
      }}).viewRow(0).viewPart(1, drmData.ncol -1)

  }

  def fit(drmData: DrmLike[K]) = {
    for (i <- 0 until macroIters) {
      partialFit(drmData)
    }
  }

  def predict(drmData: DrmLike[K]) = {
    implicit val ctx = drmData.context

    val bcArch = drmBroadcast(arch)
    val bcOffsets = drmBroadcast(offsets)
    val bcU = drmBroadcast(pv)

    implicit val ktag =  drmData.keyClassTag

    drmData.mapBlock(offsets.get(3).toInt) {
      case (keys, block: Matrix) => {
        val inCoreNetwork = new InCoreMLP()
        val localOffsets = bcOffsets.value

        inCoreNetwork.inputStart = localOffsets.get(0).toInt
        inCoreNetwork.inputOffset = localOffsets.get(1).toInt
        inCoreNetwork.targetStart = localOffsets.get(2).toInt
        inCoreNetwork.targetOffset = localOffsets.get(3).toInt

        inCoreNetwork.createWeightsArray(bcArch.value)
        inCoreNetwork.setParametersFromVector(bcU.value)

        val output = new DenseMatrix(block.nrow, inCoreNetwork.targetOffset)
        for (row <- 0 until block.nrow) {
          inCoreNetwork.feedForward(block(row, inCoreNetwork.inputStart until inCoreNetwork.inputOffset))
          output.assignRow(row, inCoreNetwork.A(inCoreNetwork.A.size -1))
        }
        (keys, output)
      }
    }
  }
}
