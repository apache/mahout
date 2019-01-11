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

package org.apache.mahout.math.algorithms.preprocessing

import collection._
import JavaConversions._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.{Vector => MahoutVector}



class MeanCenter extends PreprocessorFitter {

  /**
    * Centers Columns at zero or centers
    * @param input   A drm which to center on
    *
    */
  def fit[K](input: DrmLike[K],
             hyperparameters: (Symbol, Any)*): MeanCenterModel = {
    new MeanCenterModel(input.colMeans()) // could add centers here
  }

}

/**
  * A model for mean centering each column of a data set at 0 or some number specified by the setCenters method.
  * @param means
  */
class MeanCenterModel(means: MahoutVector) extends PreprocessorModel {

  var colCentersV: MahoutVector = means

  def setCenters(centers: MahoutVector): Unit = {
    if (means.length != centers.length){
      throw new Exception(s"Length of centers vector (${centers.length}) must equal length of means vector ((${means.length}) (e.g. the number of columns in the orignally fit input).")
    }
    colCentersV = means + centers
  }
  def transform[K](input: DrmLike[K]): DrmLike[K] = {

    implicit val ctx = input.context
    implicit val ktag =  input.keyClassTag

    val bcastV = drmBroadcast(colCentersV)

    val output = input.mapBlock(input.ncol) {
      case (keys, block: Matrix) =>
        val copy: Matrix = block.cloned
        copy.foreach(row => row -= bcastV.value)
        (keys, copy)
    }
    output
  }

  def invTransform[K](input: DrmLike[K]): DrmLike[K] = {

    implicit val ctx = input.context
    implicit val ktag =  input.keyClassTag
    val bcastV = drmBroadcast(colCentersV)

    val output = input.mapBlock(input.ncol) {
      case (keys, block: Matrix) =>
        val copy: Matrix = block.cloned
        copy.foreach(row => row += bcastV.value)
        (keys, copy)
    }
    output
  }

}