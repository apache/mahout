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

package org.apache.mahout.math.drm

import org.apache.mahout.math.indexeddataset._

import logical._
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import DistributedEngine._
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/** Abstraction of optimizer/distributed engine */
trait DistributedEngine {

  /**
   * First optimization pass. Return physical plan that we can pass to exec(). This rewrite may
   * introduce logical constructs (including engine-specific ones) that user DSL cannot even produce
   * per se.
   * <P>
   *
   * A particular physical engine implementation may choose to either use the default rewrites or
   * build its own rewriting rules.
   * <P>
   */
  def optimizerRewrite[K: ClassTag](action: DrmLike[K]): DrmLike[K] = pass3(pass2(pass1(action)))

  /** Second optimizer pass. Translate previously rewritten logical pipeline into physical engine plan. */
  def toPhysical[K: ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K]

  /** Engine-specific colSums implementation based on a checkpoint. */
  def colSums[K](drm: CheckpointedDrm[K]): Vector

  /** Optional engine-specific all reduce tensor operation. */
  def allreduceBlock[K](drm: CheckpointedDrm[K], bmf: BlockMapFunc2[K], rf: BlockReduceFunc): Matrix

  /** Engine-specific numNonZeroElementsPerColumn implementation based on a checkpoint. */
  def numNonZeroElementsPerColumn[K](drm: CheckpointedDrm[K]): Vector

  /** Engine-specific colMeans implementation based on a checkpoint. */
  def colMeans[K](drm: CheckpointedDrm[K]): Vector

  def norm[K](drm: CheckpointedDrm[K]): Double

  /** Broadcast support */
  def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector]

  /** Broadcast support */
  def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix]

  /**
   * Load DRM from hdfs (as in Mahout DRM format).
   * <P/>
   * @param path The DFS path to load from
   * @param parMin Minimum parallelism after load (equivalent to #par(min=...)).
   */
  def drmDfsRead(path: String, parMin: Int = 0)(implicit sc: DistributedContext): CheckpointedDrm[_]

  /** Parallelize in-core matrix as the backend engine distributed matrix, using row ordinal indices as data set keys. */
  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)(implicit sc: DistributedContext):
  CheckpointedDrm[Int]

  /** Parallelize in-core matrix as the backend engine distributed matrix, using row labels as a data set keys. */
  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int = 1)(implicit sc: DistributedContext):
  CheckpointedDrm[String]

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int = 10)(implicit sc: DistributedContext):
  CheckpointedDrm[Int]

  /** Creates empty DRM with non-trivial height */
  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int = 10)(implicit sc: DistributedContext):
  CheckpointedDrm[Long]

  /**
   * Convert non-int-keyed matrix to an int-keyed, computing optionally mapping from old keys
   * to row indices in the new one. The mapping, if requested, is returned as a 1-column matrix.
   */
  def drm2IntKeyed[K](drmX: DrmLike[K], computeMap: Boolean = false): (DrmLike[Int], Option[DrmLike[K]])

  /**
   * (Optional) Sampling operation. Consistent with Spark semantics of the same.
   * @param drmX
   * @param fraction
   * @param replacement
   * @tparam K
   * @return
   */
  def drmSampleRows[K](drmX: DrmLike[K], fraction: Double, replacement: Boolean = false): DrmLike[K]

  def drmSampleKRows[K](drmX: DrmLike[K], numSamples:Int, replacement:Boolean = false) : Matrix

  /**
   * Load IndexedDataset from text delimited format.
   * @param src comma delimited URIs to read from
   * @param schema defines format of file(s)
   */
  def indexedDatasetDFSRead(src: String,
      schema: Schema = DefaultIndexedDatasetReadSchema,
      existingRowIDs: Option[BiDictionary] = None)
      (implicit sc: DistributedContext):
    IndexedDataset

  /**
   * Load IndexedDataset from text delimited format, one element per line
   * @param src comma delimited URIs to read from
   * @param schema defines format of file(s)
   */
  def indexedDatasetDFSReadElements(src: String,
      schema: Schema = DefaultIndexedDatasetElementReadSchema,
      existingRowIDs: Option[BiDictionary] = None)
      (implicit sc: DistributedContext):
    IndexedDataset

}

object DistributedEngine {

  private val log = Logger.getLogger(DistributedEngine.getClass)

  /** This is mostly multiplication operations rewrites */
  private def pass1[K](action: DrmLike[K]): DrmLike[K] = {

    action match {

      // Logical but previously had checkpoint attached to it already that has some caching policy to it
      case cpa: CheckpointAction[K] if cpa.cp.exists(_.cacheHint != CacheHint.NONE) ⇒ cpa.cp.get

      // self element-wise rewrite
      case OpAewB(a, b, op) if a == b => {
        op match {
          case "*" ⇒ OpAewUnaryFunc(pass1(a), (x) ⇒ x * x)
          case "/" ⇒ OpAewUnaryFunc(pass1(a), (x) ⇒ x / x)
          // Self "+" and "-" don't make a lot of sense, but we do include it for completeness.
          case "+" ⇒ OpAewUnaryFunc(pass1(a), 2.0 * _)
          case "-" ⇒ OpAewUnaryFunc(pass1(a), (_) ⇒ 0.0)
          case _ ⇒
          require(false, s"Unsupported operator $op")
            null
        }
      }
      case OpAB(OpAt(a), b) if a == b ⇒ OpAtA(pass1(a))
      case OpABAnyKey(OpAtAnyKey(a), b) if a == b ⇒ OpAtA(pass1(a))

      // A small rule change: Now that we have removed ClassTag at the %*% operation, it doesn't
      // match b[Int] case automatically any longer. So, we need to check and rewrite it dynamically
      // and re-run pass1 again on the obtained tree.
      case OpABAnyKey(a, b) if b.keyClassTag == ClassTag.Int ⇒ pass1(OpAB(a, b.asInstanceOf[DrmLike[Int]]))
      case OpAtAnyKey(a) if a.keyClassTag == ClassTag.Int ⇒ pass1(OpAt(a.asInstanceOf[DrmLike[Int]]))

      // For now, rewrite left-multiply via transpositions, i.e.
      // inCoreA %*% B = (B' %*% inCoreA')'
      case op@OpTimesLeftMatrix(a, b) ⇒
        OpAt(OpTimesRightMatrix(A = OpAt(pass1(b)), right = a.t))

      // Add vertical row index concatenation for rbind() on DrmLike[Int] fragments
      case op@OpRbind(a, b) if op.keyClassTag == ClassTag.Int ⇒

        // Make sure closure sees only local vals, not attributes. We need to do these ugly casts
        // around because compiler could not infer that K is the same as Int, based on if() above.
        val ma = safeToNonNegInt(a.nrow)
        val bAdjusted = new OpMapBlock[Int, Int](A = pass1(b.asInstanceOf[DrmLike[Int]]), bmf = {
          case (keys, block) ⇒ keys.map(_ + ma) → block
        }, identicallyPartitioned = false)
        val aAdjusted = a.asInstanceOf[DrmLike[Int]]
        OpRbind(pass1(aAdjusted), bAdjusted).asInstanceOf[DrmLike[K]]

      // Stop at checkpoints
      case cd: CheckpointedDrm[_] ⇒ action

      // For everything else we just pass-thru the operator arguments to optimizer
      case uop: AbstractUnaryOp[_, K] ⇒
        uop.A = pass1(uop.A)
        uop

      case bop: AbstractBinaryOp[_, _, K] ⇒
        bop.A = pass1(bop.A)
        bop.B = pass1(bop.B)
        bop
    }
  }

  /** This would remove stuff like A.t.t that previous step may have created */
  private def pass2[K](action: DrmLike[K]): DrmLike[K] = {
    action match {

      // Fusion of unary funcs into single, like 1 + x * x.
      // Since we repeating the pass over self after rewrite, we dont' need to descend into arguments
      // recursively here.
      case op1@OpAewUnaryFunc(op2@OpAewUnaryFunc(a, _, _), _, _) ⇒
        pass2(OpAewUnaryFuncFusion(a, op1 :: op2 :: Nil))

      // Fusion one step further, like 1 + 2 * x * x. All should be rewritten as one UnaryFuncFusion.
      // Since we repeating the pass over self after rewrite, we dont' need to descend into arguments
      // recursively here.
      case op@OpAewUnaryFuncFusion(op2@OpAewUnaryFunc(a, _, _), _) ⇒
        pass2(OpAewUnaryFuncFusion(a, op.ff :+ op2))

      // A.t.t => A
      case OpAt(top@OpAt(a)) ⇒  pass2(a)

      // Stop at checkpoints
      case cd: CheckpointedDrm[_] ⇒  action

      // For everything else we just pass-thru the operator arguments to optimizer
      case uop: AbstractUnaryOp[_, K] ⇒
        uop.A = pass2(uop.A)
        uop
      case bop: AbstractBinaryOp[_, _, K] ⇒
        bop.A = pass2(bop.A)
        bop.B = pass2(bop.B)
        bop
    }
  }

  /** Some further rewrites that are conditioned on A.t.t removal */
  private def pass3[K](action: DrmLike[K]): DrmLike[K] = {
    action match {

      // matrix products.
      case OpAB(a, OpAt(b)) ⇒  OpABt(pass3(a), pass3(b))

      // AtB cases that make sense.
      case OpAB(OpAt(a), b) if a.partitioningTag == b.partitioningTag ⇒  OpAtB(pass3(a), pass3(b))
      case OpABAnyKey(OpAtAnyKey(a), b) ⇒  OpAtB(pass3(a), pass3(b))

      // Need some cost to choose between the following.

      case OpAB(OpAt(a), b) ⇒  OpAtB(pass3(a), pass3(b))
      //      case OpAB(OpAt(a), b) => OpAt(OpABt(OpAt(pass1(b)), pass1(a)))
      case OpAB(a, b) ⇒  OpABt(pass3(a), OpAt(pass3(b)))

      // Rewrite A'x
      case op@OpAx(op1@OpAt(a), x) ⇒  OpAtx(pass3(a), x)

      // Stop at checkpoints
      case cd: CheckpointedDrm[_] ⇒  action

      // For everything else we just pass-thru the operator arguments to optimizer
      case uop: AbstractUnaryOp[_, K] ⇒
        uop.A = pass3(uop.A)
        uop
      case bop: AbstractBinaryOp[_, _, K] ⇒
        bop.A = pass3(bop.A)
        bop.B = pass3(bop.B)
        bop
    }
  }

}