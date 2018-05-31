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

import org.apache.mahout.logging._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical.{AbstractUnaryOp, OpAewB, OpAewScalar, TEwFunc}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.sparkbindings.blas.AewB.{ReduceFunc, ReduceFuncScalar}
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.mahout.sparkbindings.{BlockifiedDrmRdd, DrmRdd, drm}

import scala.reflect.{ClassTag, classTag}
import scala.collection.JavaConversions._

/** Elementwise drm-drm operators */
object AewB {

  private final implicit val log = getLog(AewB.getClass)

  /**
   * Set to false to disallow in-place elementwise operations in case side-effects and non-idempotent
   * computations become a problem.
   */
  final val PROPERTY_AEWB_INPLACE = "mahout.math.AewB.inplace"

  type ReduceFunc = (Vector, Vector) => Vector

  type ReduceFuncScalar = (Matrix, Double) => Matrix

  private[blas] def ewInplace(): Boolean = System.getProperty(PROPERTY_AEWB_INPLACE, "false").toBoolean

  private[blas] def getEWOps() = if (ewInplace()) InplaceEWOps else CloningEWOps


  /** Elementwise matrix-matrix operator, now handles both non- and identically partitioned */
  def a_ew_b[K](op: OpAewB[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] = {

    implicit val ktag = op.keyClassTag

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

    val a = srcA.asRowWise()
    val b = srcB.asRowWise()

    debug(s"A${op.op}B: #partsA=${a.partitions.length},#partsB=${b.partitions.length}.")

    // Check if A and B are identically partitioned AND keyed. if they are, then just perform zip
    // instead of join, and apply the op map-side. Otherwise, perform join and apply the op
    // reduce-side.
    val rdd = if (op.isIdenticallyPartitioned(op.A)) {

      debug(s"A${op.op}B:applying zipped elementwise")

      a
          .zip(b)
          .map {
        case ((keyA, vectorA), (keyB, vectorB)) =>
          assert(keyA == keyB, "inputs are claimed identically partitioned, but they are not identically keyed")
          keyA -> reduceFunc(vectorA, vectorB)
      }
    } else {

      debug("A${op.op}B:applying elementwise as join")

      a
          // Full outer-join operands row-wise
          .cogroup(b, numPartitions = a.partitions.length max b.partitions.length)

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

    rdd
  }

  def a_ew_func[K](op:AbstractUnaryOp[K,K] with TEwFunc, srcA: DrmRddInput[K]):DrmRddInput[K] = {

    val evalZeros = op.evalZeros
    val inplace = ewInplace()
    val f = op.f
    implicit val ktag = op.keyClassTag

    // Before obtaining blockified rdd, see if we have to fix int row key consistency so that missing
    // rows can get lazily pre-populated with empty vectors before proceeding with elementwise scalar.
    val aBlockRdd = if (classTag[K] == ClassTag.Int && op.A.canHaveMissingRows && evalZeros) {
      val fixedRdd = fixIntConsistency(op.A.asInstanceOf[DrmLike[Int]], src = srcA.asRowWise().asInstanceOf[DrmRdd[Int]])
      drm.blockify(fixedRdd, blockncol = op.A.ncol).asInstanceOf[BlockifiedDrmRdd[K]]
    } else {
      srcA.asBlockified(op.A.ncol)
    }

    val rdd = aBlockRdd.map {case (keys, block) =>

      // Do inplace or allocate a new copy?
      val newBlock = if (inplace) block else block cloned

      // Operation cares about zeros?
      if (evalZeros) {

        // Yes, we evaluate all:
        newBlock := ((_,_,x)=>f(x))
      } else {

        // No, evaluate non-zeros only row-wise
        for (row <- newBlock; el <- row.nonZeroes) el := f(el.get)
      }

      keys -> newBlock
    }

    rdd
  }

  /** Physical algorithm to handle matrix-scalar operators like A - s or s -: A */
  def a_ew_scalar[K](op: OpAewScalar[K], srcA: DrmRddInput[K], scalar: Double):
  DrmRddInput[K] = {


    val ewOps = getEWOps()
    val opId = op.op
    implicit val ktag = op.keyClassTag

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
    val aBlockRdd = if (classTag[K] == ClassTag.Int && op.A.canHaveMissingRows) {
      val fixedRdd = fixIntConsistency(op.A.asInstanceOf[DrmLike[Int]], src = srcA.asRowWise().asInstanceOf[DrmRdd[Int]])
      drm.blockify(fixedRdd, blockncol = op.A.ncol).asInstanceOf[BlockifiedDrmRdd[K]]
    } else {
      srcA.asBlockified(op.A.ncol)
    }

    debug(s"A${op.op}$scalar: #parts=${aBlockRdd.partitions.length}.")

    val rdd = aBlockRdd
        .map {
      case (keys, block) => keys -> reduceFunc(block, scalar)
    }

    rdd
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

