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

import org.apache.log4j.Logger
import org.apache.mahout.sparkbindings.DrmRdd
import reflect._
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm.logical.{OpCbindScalar, OpCbind}

/** Physical cbind */
object CbindAB {

  private val log = Logger.getLogger(CbindAB.getClass)

  def cbindAScalar[K](op: OpCbindScalar[K], srcA:DrmRddInput[K]) : DrmRddInput[K] = {

    implicit val ktag = op.keyClassTag
    val srcRdd = srcA.asRowWise()

    val ncol = op.A.ncol
    val x = op.x

    val fixedRdd = if (classTag[K] == ClassTag.Int && x != 0.0)
      fixIntConsistency(op.asInstanceOf[OpCbindScalar[Int]],
        src = srcRdd.asInstanceOf[DrmRdd[Int]]).asInstanceOf[DrmRdd[K]]
    else srcRdd

    val left = op.leftBind

    val resultRdd = fixedRdd.map { case (key, vec) =>
      val newVec = vec.like(ncol + 1)
      if (left) {
        newVec(1 to ncol) := vec
        newVec(0) = x
      } else {
        newVec(0 until ncol) := vec
        newVec(ncol) = x
      }
      key -> newVec
    }

    resultRdd
  }

  def cbindAB_nograph[K](op: OpCbind[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] = {

    val a = srcA.asRowWise()
    val b = srcB.asRowWise()
    val n = op.ncol
    val n1 = op.A.ncol
    val n2 = n - n1
    implicit val ktag = op.keyClassTag

    // Check if A and B are identically partitioned AND keyed. if they are, then just perform zip
    // instead of join, and apply the op map-side. Otherwise, perform join and apply the op
    // reduce-side.
    val rdd = if (op.isIdenticallyPartitioned(op.A)) {

      log.debug("applying zipped cbind()")

      a
          .zip(b)
          .map {
        case ((keyA, vectorA), (keyB, vectorB)) =>
          assert(keyA == keyB, "inputs are claimed identically partitioned, but they are not identically keyed")

          val dense = vectorA.isDense && vectorB.isDense
          val vec: Vector = if (dense) new DenseVector(n) else new SequentialAccessSparseVector(n)
          vec(0 until n1) := vectorA
          vec(n1 until n) := vectorB
          keyA -> vec
      }
    } else {

      log.debug("applying cbind as join")

      a
          .cogroup(b, numPartitions = a.partitions.length max b.partitions.length)
          .map {
        case (key, (vectorSeqA, vectorSeqB)) =>

          // Generally, after co-grouping, we should not accept anything but 1 to 1 in the left and
          // the right groups. However let's be flexible here, if it does happen, recombine them into 1.

          val vectorA = if (vectorSeqA.size <= 1)
            vectorSeqA.headOption.getOrElse(new RandomAccessSparseVector(n1))
          else
            (vectorSeqA.head.like() /: vectorSeqA)(_ += _)

          val vectorB = if ( vectorSeqB.size <= 1)
            vectorSeqB.headOption.getOrElse(new RandomAccessSparseVector(n2))
          else
            (vectorSeqB.head.like() /: vectorSeqB)(_ += _)

          val dense = vectorA.isDense && vectorB.isDense
          val vec:Vector = if (dense) new DenseVector(n) else new SequentialAccessSparseVector(n)
          vec(0 until n1) := vectorA
          vec(n1 until n) := vectorB
          key -> vec
      }
    }

    rdd

  }

}
