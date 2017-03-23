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
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.{Vector => MahoutVector, Matrix}

/**
  * Scales columns to mean 0 and unit variance
  *
  * An important note- The equivelent call in R would be something like
  * ```r
  * N <- nrow(x)
  * scale(x, scale= apply(x, 2, sd) * sqrt(N-1/N))
  * ```
  *
  * This is because R uses degrees of freedom = 1 to calculate standard deviation.
  * Multiplying the standard deviation by sqrt(N-1/N) 'undoes' this correction.
  *
  * The StandardScaler of sklearn uses degrees of freedom = 0 for its calculation, so results
  * should be similar.
  */
class StandardScaler extends PreprocessorFitter {

  def fit[K](input: DrmLike[K],
             hyperparameters: (Symbol, Any)*): StandardScalerModel = {
    val mNv = dcolMeanVars(input)
    new StandardScalerModel(mNv._1, mNv._2.sqrt)
  }

}

class StandardScalerModel(val meanVec: MahoutVector,
                          val stdev: MahoutVector
                         ) extends PreprocessorModel {


  def transform[K](input: DrmLike[K]): DrmLike[K] = {
    implicit val ctx = input.context

    // Some mapBlock() calls need it
    // implicit val ktag =  input.keyClassTag

    val bcastMu = drmBroadcast(meanVec)
    val bcastSigma = drmBroadcast(stdev)

    implicit val ktag =  input.keyClassTag

    val res = input.mapBlock(input.ncol) {
      case (keys, block: Matrix) => {
        val copy: Matrix = block.cloned
        copy.foreach(row => row := (row - bcastMu) / bcastSigma )
        (keys, copy)
      }
    }
    res
  }

  /**
    * Given a an output- trasform it back into the original
    * e.g. a normalized column, back to original values.
    *
    * @param input
    * @tparam K
    * @return
    */
  def invTransform[K](input: DrmLike[K]): DrmLike[K] = { // [K: ClassTag]

    implicit val ctx = input.context

    // Some mapBlock() calls need it
    implicit val ktag =  input.keyClassTag

    val bcastMu = drmBroadcast(meanVec)
    val bcastSigma = drmBroadcast(stdev)

    val res = input.mapBlock(input.ncol) {
      case (keys, block: Matrix) => {
        val copy: Matrix = block.cloned
        copy.foreach(row => row := (row * bcastSigma ) + bcastMu)
        (keys, copy)
      }
    }
    res
  }
}