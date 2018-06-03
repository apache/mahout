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
package org.apache.mahout.sparkbindings.blas

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import scala.reflect.ClassTag
import org.apache.mahout.math.drm.logical.{OpAx, OpAtx}


/** Matrix product with one of operands an in-core matrix */
object Ax {

  def ax_with_broadcast[K](op: OpAx[K], srcA: DrmRddInput[K]): DrmRddInput[K] = {

    val rddA = srcA.asBlockified(op.A.ncol)
    implicit val sc: DistributedContext = rddA.sparkContext
    implicit val ktag = op.keyClassTag

    val bcastX = drmBroadcast(op.x)

    val rdd: BlockifiedDrmRdd[K] = rddA

      // Just multiply the blocks
      .map { case (keys, blockA) ⇒ keys → (blockA %*% bcastX).toColMatrix }

    new DrmRddInput(Right(rdd))
  }

  def atx_with_broadcast(op: OpAtx, srcA: DrmRddInput[Int]): DrmRddInput[Int] = {

    val rddA = srcA.asBlockified(op.A.ncol)
    implicit val dc:DistributedContext = rddA.sparkContext

    val bcastX = drmBroadcast(op.x)

    val inCoreM = rddA
        // Just multiply the blocks
        .map {
      case (keys, blockA) =>
        keys.zipWithIndex.map {
          case (key, idx) => blockA(idx, ::) * bcastX.value(key)
        }
            .reduce(_ += _)
    }
        // All-reduce
        .reduce(_ += _)
        // Convert back to mtx
        .toColMatrix

    // It is ridiculous, but in this scheme we will have to re-parallelize it again in order to plug
    // it back as drm blockified rdd

    val rdd:BlockifiedDrmRdd[Int] = dc.parallelize(Seq(inCoreM), numSlices = 1)
        .map{block ⇒ Array.tabulate(block.nrow)(i ⇒ i) -> block}

    rdd

  }

}
