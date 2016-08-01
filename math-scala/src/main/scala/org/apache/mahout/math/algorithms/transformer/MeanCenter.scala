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

package org.apache.mahout.math.algorithms.transformer

import collection._
import JavaConversions._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.reflect.ClassTag

class MeanCenter extends Transformer {

  /**
    * Optionally set the centers of each column to some value other than Zero
    * @param centers A vector of length equal to the `input` in the fit method specifying the
    *                centers to set each column to.
    */
  def setCenter(centers: Vector) = {
    fitParams("colMeansV") = fitParams("colMeansV") - centers
  }

  /**
    * Centers Columns at zero
    * @param input
    * @tparam Int
    */
  def fit[Int](input: DrmLike[Int]) = {
    fitParams("colMeansV") = input.colMeans
  }

  def transform[Int: ClassTag](input: DrmLike[Int]): DrmLike[Int] = {

    if (!isFit) {
      //throw an error
    }

    implicit val ctx = input.context
    val colMeansV = fitParams.get("colMeansV").get
    val bcastV = drmBroadcast(colMeansV)

    val output = input.mapBlock(input.ncol) {
      case (keys, block) =>
        val copy: Matrix = block.cloned
        copy.foreach(row => row -= bcastV.value)
        (keys, copy)
    }
    output
  }

  def invTransform[K: ClassTag](input: DrmLike[K]): DrmLike[K] = {

    if (!isFit) {
      //throw an error
    }

    implicit val ctx = input.context
    val colMeansV = fitParams.get("colMeansV").get
    val bcastV = drmBroadcast(colMeansV)

    val output = input.mapBlock(input.ncol) {
      case (keys, block) =>
        val copy: Matrix = block.cloned
        copy.foreach(row => row += bcastV.value)
        (keys, copy)
    }
    output
  }

  def summary(): String = {
    "not implemented yet"
  }

}
