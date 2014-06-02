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

import org.apache.mahout.sparkbindings.drm.DrmRddInput
import scala.reflect.ClassTag
import org.apache.spark.SparkContext._
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.drm.logical.{OpAewScalar, OpAewB}
import org.apache.log4j.Logger

/** Elementwise drm-drm operators */
object AewB {


  private val log = Logger.getLogger(AewB.getClass)

  @inline
  def a_plus_b[K: ClassTag](op: OpAewB[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] =
    a_ew_b(op, srcA, srcB, reduceFunc = (a, b) => a += b)

  @inline
  def a_minus_b[K: ClassTag](op: OpAewB[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] =
    a_ew_b(op, srcA, srcB, reduceFunc = (a, b) => a -= b)

  @inline
  def a_hadamard_b[K: ClassTag](op: OpAewB[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] =
    a_ew_b(op, srcA, srcB, reduceFunc = (a, b) => a *= b)

  @inline
  def a_eldiv_b[K: ClassTag](op: OpAewB[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] =
    a_ew_b(op, srcA, srcB, reduceFunc = (a, b) => a /= b)

  @inline
  def a_plus_scalar[K: ClassTag](op: OpAewScalar[K], srcA: DrmRddInput[K], scalar: Double): DrmRddInput[K] =
    a_ew_scalar(op, srcA, scalar, reduceFunc = (A, s) => A += s)

  @inline
  def a_minus_scalar[K: ClassTag](op: OpAewScalar[K], srcA: DrmRddInput[K], scalar: Double): DrmRddInput[K] =
    a_ew_scalar(op, srcA, scalar, reduceFunc = (A, s) => A -= s)

  @inline
  def scalar_minus_a[K: ClassTag](op: OpAewScalar[K], srcA: DrmRddInput[K], scalar: Double): DrmRddInput[K] =
    a_ew_scalar(op, srcA, scalar, reduceFunc = (A, s) => s -=: A)

  @inline
  def a_times_scalar[K: ClassTag](op: OpAewScalar[K], srcA: DrmRddInput[K], scalar: Double): DrmRddInput[K] =
    a_ew_scalar(op, srcA, scalar, reduceFunc = (A, s) => A *= s)

  @inline
  def a_div_scalar[K: ClassTag](op: OpAewScalar[K], srcA: DrmRddInput[K], scalar: Double): DrmRddInput[K] =
    a_ew_scalar(op, srcA, scalar, reduceFunc = (A, s) => A /= s)

  @inline
  def scalar_div_a[K: ClassTag](op: OpAewScalar[K], srcA: DrmRddInput[K], scalar: Double): DrmRddInput[K] =
    a_ew_scalar(op, srcA, scalar, reduceFunc = (A, s) => s /=: A)

  /** Parallel way of this operation (this assumes different partitioning of the sources */
  private[blas] def a_ew_b[K: ClassTag](op: OpAewB[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K],
      reduceFunc: (Vector, Vector) => Vector): DrmRddInput[K] = {
    val a = srcA.toDrmRdd()
    val b = srcB.toDrmRdd()

    // Check if A and B are identically partitioned AND keyed. if they are, then just perform zip
    // instead of join, and apply the op map-side. Otherwise, perform join and apply the op
    // reduce-side.
    val rdd = if (op.isIdenticallyPartitioned(op.A)) {

      log.debug("applying zipped elementwise")

      a
          .zip(b)
          .map {
        case ((keyA, vectorA), (keyB, vectorB)) =>
          assert(keyA == keyB, "inputs are claimed identically partitioned, but they are not identically keyed")

          printf("A=%s,B=%s\n",vectorA,vectorB)
          val v = vectorA + vectorB //reduceFunc(vectorA,vectorB)
          printf("A+B=%s\n",v)

          keyA -> v //reduceFunc(vectorA, vectorB)
      }
    } else {

      log.debug("applying elementwise as join")

      a
          .cogroup(b, numPartitions = a.partitions.size max b.partitions.size)
          .map({
        case (key, (vectorSeqA, vectorSeqB)) =>
          key -> reduceFunc(vectorSeqA.reduce(reduceFunc), vectorSeqB.reduce(reduceFunc))
      })
    }

    new DrmRddInput(rowWiseSrc = Some(op.ncol -> rdd))
  }

  private[blas] def a_ew_scalar[K: ClassTag](op: OpAewScalar[K], srcA: DrmRddInput[K], scalar: Double,
      reduceFunc: (Matrix, Double) => Matrix): DrmRddInput[K] = {
    val a = srcA.toBlockifiedDrmRdd()
    val rdd = a
        .map({
      case (keys, block) => keys -> reduceFunc(block, scalar)
    })
    new DrmRddInput[K](blockifiedSrc = Some(rdd))
  }


}
