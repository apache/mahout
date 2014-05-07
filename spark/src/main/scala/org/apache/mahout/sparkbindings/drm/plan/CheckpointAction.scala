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

package org.apache.mahout.sparkbindings.drm.plan

import org.apache.mahout.math.scalabindings._
import RLikeOps._
import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings.blas._
import org.apache.mahout.sparkbindings.drm._
import CheckpointAction._
import org.apache.spark.SparkContext._
import org.apache.hadoop.io.Writable
import org.apache.spark.storage.StorageLevel
import scala.util.Random

/** Implementation of distributed expression checkpoint and optimizer. */
abstract class CheckpointAction[K: ClassTag] extends DrmLike[K] {

  private[sparkbindings] lazy val partitioningTag: Long = Random.nextLong()

  private var cp:Option[CheckpointedDrm[K]] = None


  def isIdenticallyPartitioned(other:DrmLike[_]) =
    partitioningTag!= 0L && partitioningTag == other.partitioningTag

  /**
   * Action operator -- does not necessary means Spark action; but does mean running BLAS optimizer
   * and writing down Spark graph lineage since last checkpointed DRM.
   */
  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = cp.getOrElse({
    // Non-zero count is sparsely supported by logical operators now. So assume we have no knowledge
    // if it is unsupported, instead of failing.
    val plan = optimize(this)
    val rdd = exec(plan)
    val newcp = new CheckpointedDrmBase(
      rdd = rdd,
      _nrow = nrow,
      _ncol = ncol,
      _cacheStorageLevel = cacheHint2Spark(cacheHint),
      partitioningTag = plan.partitioningTag
    )
    cp = Some(newcp)
    newcp.cache()
  })

}

object CheckpointAction {

  /** Perform expression optimization. Return physical plan that we can pass to exec() */
  def optimize[K: ClassTag](action: DrmLike[K]): DrmLike[K] = pass3(pass2(pass1(action)))

  private def cacheHint2Spark(cacheHint: CacheHint.CacheHint): StorageLevel = cacheHint match {
    case CacheHint.NONE => StorageLevel.NONE
    case CacheHint.DISK_ONLY => StorageLevel.DISK_ONLY
    case CacheHint.DISK_ONLY_2 => StorageLevel.DISK_ONLY_2
    case CacheHint.MEMORY_ONLY => StorageLevel.MEMORY_ONLY
    case CacheHint.MEMORY_ONLY_2 => StorageLevel.MEMORY_ONLY_2
    case CacheHint.MEMORY_ONLY_SER => StorageLevel.MEMORY_ONLY_SER
    case CacheHint.MEMORY_ONLY_SER_2 => StorageLevel.MEMORY_ONLY_SER_2
    case CacheHint.MEMORY_AND_DISK => StorageLevel.MEMORY_AND_DISK
    case CacheHint.MEMORY_AND_DISK_2 => StorageLevel.MEMORY_AND_DISK_2
    case CacheHint.MEMORY_AND_DISK_SER => StorageLevel.MEMORY_AND_DISK_SER
    case CacheHint.MEMORY_AND_DISK_SER_2 => StorageLevel.MEMORY_AND_DISK_SER_2
  }


  /** This is mostly multiplication operations rewrites */
  private def pass1[K: ClassTag](action: DrmLike[K]): DrmLike[K] = {

    action match {
      case OpAB(OpAt(a), b) if (a == b) => OpAtA(pass1(a))
      case OpABAnyKey(OpAtAnyKey(a), b) if (a == b) => OpAtA(pass1(a))

      // For now, rewrite left-multiply via transpositions, i.e.
      // inCoreA %*% B = (B' %*% inCoreA')'
      case op@OpTimesLeftMatrix(a, b) =>
        OpAt(OpTimesRightMatrix(A = OpAt(pass1(b)), right = a.t))

      // Stop at checkpoints
      case cd: CheckpointedDrm[_] => action

      // For everything else we just pass-thru the operator arguments to optimizer
      case uop: AbstractUnaryOp[_, K] =>
        uop.A = pass1(uop.A)(uop.classTagA)
        uop
      case bop: AbstractBinaryOp[_, _, K] =>
        bop.A = pass1(bop.A)(bop.classTagA)
        bop.B = pass1(bop.B)(bop.classTagB)
        bop
    }
  }

  /** This would remove stuff like A.t.t that previous step may have created */
  private def pass2[K: ClassTag](action: DrmLike[K]): DrmLike[K] = {
    action match {
      // A.t.t => A
      case OpAt(top@OpAt(a)) => pass2(a)(top.classTagA)

      // A.t.t => A
//      case OpAt(top@OpAtAnyKey(a)) => pass2(a)(top.classTagA)


      // Stop at checkpoints
      case cd: CheckpointedDrm[_] => action

      // For everything else we just pass-thru the operator arguments to optimizer
      case uop: AbstractUnaryOp[_, K] =>
        uop.A = pass2(uop.A)(uop.classTagA)
        uop
      case bop: AbstractBinaryOp[_, _, K] =>
        bop.A = pass2(bop.A)(bop.classTagA)
        bop.B = pass2(bop.B)(bop.classTagB)
        bop
    }
  }

 /** Some further rewrites that are conditioned on A.t.t removal */
  private def pass3[K: ClassTag](action: DrmLike[K]): DrmLike[K] = {
    action match {

      // matrix products.
      case OpAB(a, OpAt(b)) => OpABt(pass3(a), pass3(b))

      // AtB cases that make sense.
      case OpAB(OpAt(a), b) if (a.partitioningTag == b.partitioningTag) => OpAtB(pass3(a), pass3(b))
      case OpABAnyKey(OpAtAnyKey(a), b) => OpAtB(pass3(a), pass3(b))

      // Need some cost to choose between the following.

      case OpAB(OpAt(a), b) => OpAtB(pass3(a), pass3(b))
      //      case OpAB(OpAt(a), b) => OpAt(OpABt(OpAt(pass1(b)), pass1(a)))
      case OpAB(a, b) => OpABt(pass3(a), OpAt(pass3(b)))
      // Rewrite A'x
      case op@OpAx(op1@OpAt(a), x) => OpAtx(pass3(a)(op1.classTagA), x)

      // Stop at checkpoints
      case cd: CheckpointedDrm[_] => action

      // For everything else we just pass-thru the operator arguments to optimizer
      case uop: AbstractUnaryOp[_, K] =>
        uop.A = pass3(uop.A)(uop.classTagA)
        uop
      case bop: AbstractBinaryOp[_, _, K] =>
        bop.A = pass3(bop.A)(bop.classTagA)
        bop.B = pass3(bop.B)(bop.classTagB)
        bop
    }
  }


  /** Execute previously optimized physical plan */
  def exec[K: ClassTag](oper: DrmLike[K]): DrmRddInput[K] = {
    // I do explicit evidence propagation here since matching via case classes seems to be loosing
    // it and subsequently may cause something like DrmRddInput[Any] instead of [Int] or [String].
    // Hence you see explicit evidence attached to all recursive exec() calls.
    oper match {
      // If there are any such cases, they must go away in pass1. If they were not, then it wasn't
      // the A'A case but actual transposition intent which should be removed from consideration
      // (we cannot do actual flip for non-int-keyed arguments)
      case OpAtAnyKey(_) =>
        throw new IllegalArgumentException("\"A\" must be Int-keyed in this A.t expression.")
      case op@OpAt(a) => At.at(op, exec(a)(op.classTagA))
      case op@OpABt(a, b) => ABt.abt(op, exec(a)(op.classTagA), exec(b)(op.classTagB))
      case op@OpAtB(a, b) => AtB.atb_nograph(op, exec(a)(op.classTagA), exec(b)(op.classTagB),
        zippable = a.partitioningTag == b.partitioningTag)
      case op@OpAtA(a) => AtA.at_a(op, exec(a)(op.classTagA))
      case op@OpAx(a, x) => Ax.ax_with_broadcast(op, exec(a)(op.classTagA))
      case op@OpAtx(a, x) => Ax.atx_with_broadcast(op, exec(a)(op.classTagA))
      case op@OpAewB(a, b, '+') => AewB.a_plus_b(op, exec(a)(op.classTagA), exec(b)(op.classTagB))
      case op@OpAewB(a, b, '-') => AewB.a_minus_b(op, exec(a)(op.classTagA), exec(b)(op.classTagB))
      case op@OpAewB(a, b, '*') => AewB.a_hadamard_b(op, exec(a)(op.classTagA), exec(b)(op.classTagB))
      case op@OpAewB(a, b, '/') => AewB.a_eldiv_b(op, exec(a)(op.classTagA), exec(b)(op.classTagB))
      case op@OpAewScalar(a, s, "+") => AewB.a_plus_scalar(op, exec(a)(op.classTagA), s)
      case op@OpAewScalar(a, s, "-") => AewB.a_minus_scalar(op, exec(a)(op.classTagA), s)
      case op@OpAewScalar(a, s, "-:") => AewB.scalar_minus_a(op, exec(a)(op.classTagA), s)
      case op@OpAewScalar(a, s, "*") => AewB.a_times_scalar(op, exec(a)(op.classTagA), s)
      case op@OpAewScalar(a, s, "/") => AewB.a_div_scalar(op, exec(a)(op.classTagA), s)
      case op@OpAewScalar(a, s, "/:") => AewB.scalar_div_a(op, exec(a)(op.classTagA), s)
      case op@OpRowRange(a, _) => Slicing.rowRange(op, exec(a)(op.classTagA))
      case op@OpTimesRightMatrix(a, _) => AinCoreB.rightMultiply(op, exec(a)(op.classTagA))
      // Custom operators, we just execute them
      case blockOp: OpMapBlock[K, _] => blockOp.exec(src = exec(blockOp.A)(blockOp.classTagA))
      case cp: CheckpointedDrm[K] => new DrmRddInput[K](rowWiseSrc = Some((cp.ncol, cp.rdd)))
      case _ => throw new IllegalArgumentException("Internal:Optimizer has no exec policy for operator %s."
          .format(oper))

    }
  }

}
