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
import org.apache.mahout.math.{SequentialAccessSparseVector, Matrix, Vector}
import org.apache.mahout.math.drm.logical.{OpAewScalar, OpAewB}
import org.apache.log4j.Logger
import org.apache.mahout.sparkbindings.blas.AewB.{ReduceFuncScalar, ReduceFunc}
import org.apache.mahout.sparkbindings.{BlockifiedDrmRdd, DrmRdd, drm}
import org.apache.mahout.math.drm._

/** Elementwise drm-drm operators */
object AewB {

  private val log = Logger.getLogger(AewB.getClass)

  /**
   * Set to false to disallow in-place elementwise operations in case side-effects and non-idempotent
   * computations become a problem.
   */
  final val PROPERTY_AEWB_INPLACE = "mahout.math.AewB.inplace"

  type ReduceFunc = (Vector, Vector) => Vector

  type ReduceFuncScalar = (Matrix, Double) => Matrix

  private[blas] def getEWOps() = {
    val inplaceProp = System.getProperty(PROPERTY_AEWB_INPLACE, "true").toBoolean
    if (inplaceProp) InplaceEWOps else CloningEWOps
  }

  /** Elementwise matrix-matrix operator, now handles both non- and identically partitioned */
  def a_ew_b[K: ClassTag](op: OpAewB[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] = {

    val ewOps = getEWOps()
    val opId = op.op
    val ncol = op.ncol

    val reduceFunc = opId match {
      case "+" => ewOps.plus
      case "-" => ewOps.minus
      case "*" => ewOps.hadamard
      case "/" => ewOps.eldiv
      case default => throw new IllegalArgumentException("Unsupported elementwise operator:%s.".format(opId))
    }

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
          keyA -> reduceFunc(vectorA, vectorB)
      }
    } else {

      log.debug("applying elementwise as join")

      a
          // Full outer-join operands row-wise
          .cogroup(b, numPartitions = a.partitions.size max b.partitions.size)

          // Reduce both sides. In case there are duplicate rows in RHS or LHS, they are summed up
          // prior to reduction.
          .map({
        case (key, (vectorSeqA, vectorSeqB)) =>
          val lhsVec: Vector = if (vectorSeqA.isEmpty) new SequentialAccessSparseVector(ncol)
          else
            (vectorSeqA.head /: vectorSeqA.tail)(_ += _)
          val rhsVec: Vector = if (vectorSeqB.isEmpty) new SequentialAccessSparseVector(ncol)
          else
            (vectorSeqB.head /: vectorSeqB.tail)(_ += _)
          key -> reduceFunc(lhsVec, rhsVec)
      })
    }

    new DrmRddInput(rowWiseSrc = Some(ncol -> rdd))
  }

  /** Physical algorithm to handle matrix-scalar operators like A - s or s -: A */
  def a_ew_scalar[K: ClassTag](op: OpAewScalar[K], srcA: DrmRddInput[K], scalar: Double):
  DrmRddInput[K] = {

    val ewOps = getEWOps()
    val opId = op.op

    val reduceFunc = opId match {
      case "+" => ewOps.plusScalar
      case "-" => ewOps.minusScalar
      case "*" => ewOps.timesScalar
      case "/" => ewOps.divScalar
      case "-:" => ewOps.scalarMinus
      case "/:" => ewOps.scalarDiv
      case default => throw new IllegalArgumentException("Unsupported elementwise operator:%s.".format(opId))
    }

    // Before obtaining blockified rdd, see if we have to fix int row key consistency so that missing 
    // rows can get lazily pre-populated with empty vectors before proceeding with elementwise scalar.
    val aBlockRdd = if (implicitly[ClassTag[K]] == ClassTag.Int && op.A.canHaveMissingRows) {
      val fixedRdd = fixIntConsistency(op.A.asInstanceOf[DrmLike[Int]], src = srcA.toDrmRdd().asInstanceOf[DrmRdd[Int]])
      drm.blockify(fixedRdd, blockncol = op.A.ncol).asInstanceOf[BlockifiedDrmRdd[K]]
    } else {
      srcA.toBlockifiedDrmRdd()
    }

    val rdd = aBlockRdd
        .map({
      case (keys, block) => keys -> reduceFunc(block, scalar)
    })

    new DrmRddInput[K](blockifiedSrc = Some(rdd))
  }
}

trait EWOps {

  val plus: ReduceFunc

  val minus: ReduceFunc

  val hadamard: ReduceFunc

  val eldiv: ReduceFunc

  val plusScalar: ReduceFuncScalar

  val minusScalar: ReduceFuncScalar

  val scalarMinus: ReduceFuncScalar

  val timesScalar: ReduceFuncScalar

  val divScalar: ReduceFuncScalar

  val scalarDiv: ReduceFuncScalar

}

object InplaceEWOps extends EWOps {
  val plus: ReduceFunc = (a, b) => a += b
  val minus: ReduceFunc = (a, b) => a -= b
  val hadamard: ReduceFunc = (a, b) => a *= b
  val eldiv: ReduceFunc = (a, b) => a /= b
  val plusScalar: ReduceFuncScalar = (A, s) => A += s
  val minusScalar: ReduceFuncScalar = (A, s) => A -= s
  val scalarMinus: ReduceFuncScalar = (A, s) => s -=: A
  val timesScalar: ReduceFuncScalar = (A, s) => A *= s
  val divScalar: ReduceFuncScalar = (A, s) => A /= s
  val scalarDiv: ReduceFuncScalar = (A, s) => s /=: A
}

object CloningEWOps extends EWOps {
  val plus: ReduceFunc = (a, b) => a + b
  val minus: ReduceFunc = (a, b) => a - b
  val hadamard: ReduceFunc = (a, b) => a * b
  val eldiv: ReduceFunc = (a, b) => a / b
  val plusScalar: ReduceFuncScalar = (A, s) => A + s
  val minusScalar: ReduceFuncScalar = (A, s) => A - s
  val scalarMinus: ReduceFuncScalar = (A, s) => s -: A
  val timesScalar: ReduceFuncScalar = (A, s) => A * s
  val divScalar: ReduceFuncScalar = (A, s) => A / s
  val scalarDiv: ReduceFuncScalar = (A, s) => s /: A
}

