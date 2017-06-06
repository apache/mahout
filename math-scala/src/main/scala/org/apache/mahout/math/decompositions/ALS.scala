/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math.decompositions

import org.apache.mahout.math._
import drm._
import scalabindings._
import RLikeDrmOps._
import RLikeOps._
import org.apache.log4j.Logger
import math._
import org.apache.mahout.common.RandomUtils

/** Simple ALS factorization algotithm. To solve, use train() method. */
private[math] object ALS {

  private val log = Logger.getLogger(ALS.getClass)

  /**
   * ALS training result. <P>
   *
   * <code>drmU %*% drmV.t</code> is supposed to approximate the input.
   *
   * @param drmU U matrix
   * @param drmV V matrix
   * @param iterationsRMSE RMSE values afeter each of iteration performed
   */
  class Result[K](val drmU: DrmLike[K], val drmV: DrmLike[Int], val iterationsRMSE: Iterable[Double]) {
    def toTuple = (drmU, drmV, iterationsRMSE)
  }

  /** Result class for in-core results */
  class InCoreResult(val inCoreU: Matrix, inCoreV: Matrix, val iterationsRMSE: Iterable[Double]) {
    def toTuple = (inCoreU, inCoreV, iterationsRMSE)
  }

  /**
   * Run Distributed ALS.
   * <P>
   *
   * Example:
   *
   * <pre>
   * val (u,v,errors) = als(input, k).toTuple
   * </pre>
   *
   * ALS runs until (rmse[i-1]-rmse[i])/rmse[i-1] < convergenceThreshold, or i==maxIterations,
   * whichever earlier.
   * <P>
   *
   * @param drmA The input matrix
   * @param k required rank of decomposition (number of cols in U and V results)
   * @param convergenceThreshold stop sooner if (rmse[i-1] - rmse[i])/rmse[i - 1] is less than this
   *                             value. If <=0 then we won't compute RMSE and use convergence test.
   * @param lambda regularization rate
   * @param maxIterations maximum iterations to run regardless of convergence
   * @tparam K row key type of the input (100 is probably more than enough)
   * @return { @link org.apache.mahout.math.drm.decompositions.ALS.Result}
   */
  def dals[K](
      drmA: DrmLike[K],
      k: Int = 50,
      lambda: Double = 0.0,
      maxIterations: Int = 10,
      convergenceThreshold: Double = 0.10
      ): Result[K] = {

    assert(convergenceThreshold < 1.0, "convergenceThreshold")
    assert(maxIterations >= 1, "maxIterations")

    // Some mapblock() usage may require to know ClassTag[K] bound
    implicit val ktag = drmA.keyClassTag

    val drmAt = drmA.t

    // Initialize U and V so that they are identically distributed to A or A'
    var drmU = drmA.mapBlock(ncol = k) {
      case (keys, block) =>
        val rnd = RandomUtils.getRandom()
        val uBlock = Matrices.symmetricUniformView(block.nrow, k, rnd.nextInt()) * 0.01
        keys -> uBlock
    }

    var drmV: DrmLike[Int] = null
    var rmseIterations: List[Double] = Nil

    // ALS iterator
    var stop = false
    var i = 0
    while (!stop && i < maxIterations) {

      // Alternate. This is really what ALS is.
      if (drmV != null) drmV.uncache()
      drmV = (drmAt %*% drmU %*% solve(drmU.t %*% drmU -: diag(lambda, k))).checkpoint()

      drmU.uncache()
      drmU = (drmA %*% drmV %*% solve(drmV.t %*% drmV -: diag(lambda, k))).checkpoint()

      // Check if we are requested to do a convergence test; and do it if yes.
      if (convergenceThreshold > 0) {

        val rmse = (drmA - drmU %*% drmV.t).norm / sqrt(drmA.ncol * drmA.nrow)

        if (i > 0) {
          val rmsePrev = rmseIterations.last
          val convergence = (rmsePrev - rmse) / rmsePrev

          if (convergence < 0) {
            log.warn("Rmse increase of %f. Should not happen.".format(convergence))
            // I guess error growth can happen in ideal data case?
            stop = true
          } else if (convergence < convergenceThreshold) {
            stop = true
          }
        }
        rmseIterations :+= rmse
      }

      i += 1
    }

    new Result(drmU, drmV, rmseIterations)
  }


}
