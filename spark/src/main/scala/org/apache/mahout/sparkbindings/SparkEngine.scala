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

package org.apache.mahout.sparkbindings

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.sparkbindings.drm.{CheckpointedDrmSpark, DrmRddInput}
import org.apache.mahout.math._
import scala.reflect.ClassTag
import org.apache.spark.storage.StorageLevel
import org.apache.mahout.sparkbindings.blas._
import org.apache.hadoop.io.{LongWritable, Text, IntWritable, Writable}
import scala.Some
import scala.collection.JavaConversions._
import org.apache.spark.SparkContext
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.spark.rdd.RDD

/** Spark-specific non-drm-method operations */
object SparkEngine extends DistributedEngine {

  def colSums[K:ClassTag](drm: CheckpointedDrm[K]): Vector = {
    val n = drm.ncol

    drm.rdd
      // Throw away keys
      .map(_._2)
      // Fold() doesn't work with kryo still. So work around it.
      .mapPartitions(iter => {
      val acc = ((new DenseVector(n): Vector) /: iter)((acc, v) => acc += v)
      Iterator(acc)
    })
      // Since we preallocated new accumulator vector per partition, this must not cause any side
      // effects now.
      .reduce(_ += _)
  }

  def numNonZeroElementsPerColumn[K:ClassTag](drm: CheckpointedDrm[K]): Vector = {
    val n = drm.ncol

    drm.rdd
      // Throw away keys
      .map(_._2)
      // Fold() doesn't work with kryo still. So work around it.
      .mapPartitions(iter => {
      val acc = ((new DenseVector(n): Vector) /: iter) { (acc, v) =>
        v.nonZeroes().foreach { elem => acc(elem.index) += 1}
        acc
      }
      Iterator(acc)
    })
      // Since we preallocated new accumulator vector per partition, this must not cause any side
      // effects now.
      .reduce(_ += _)
  }

  /** Engine-specific colMeans implementation based on a checkpoint. */
  override def colMeans[K:ClassTag](drm: CheckpointedDrm[K]): Vector =
    if (drm.nrow == 0) drm.colSums() else drm.colSums() /= drm.nrow

  override def norm[K: ClassTag](drm: CheckpointedDrm[K]): Double =
    drm.rdd
        // Compute sum of squares of each vector
        .map {
      case (key, v) => v dot v
    }
        .reduce(_ + _)


  /**
   * Perform default expression rewrite. Return physical plan that we can pass to exec(). <P>
   *
   * A particular physical engine implementation may choose to either use or not use these rewrites
   * as a useful basic rewriting rule.<P>
   */
  override def optimizerRewrite[K: ClassTag](action: DrmLike[K]): DrmLike[K] = super.optimizerRewrite(action)


  /** Second optimizer pass. Translate previously rewritten logical pipeline into physical engine plan. */
  def toPhysical[K: ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] = {

    // Spark-specific Physical Plan translation.
    val rdd = tr2phys(plan)

    val newcp = new CheckpointedDrmSpark(
      rdd = rdd,
      _nrow = plan.nrow,
      _ncol = plan.ncol,
      _cacheStorageLevel = cacheHint2Spark(ch),
      partitioningTag = plan.partitioningTag
    )
    newcp.cache()
  }

  /** Broadcast support */
  def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] = dc.broadcast(v)

  /** Broadcast support */
  def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] = dc.broadcast(m)

  /**
   * Load DRM from hdfs (as in Mahout DRM format)
   *
   * @param path
   * @param sc spark context (wanted to make that implicit, doesn't work in current version of
   *           scala with the type bounds, sorry)
   *
   * @return DRM[Any] where Any is automatically translated to value type
   */
  def drmFromHDFS (path: String, parMin:Int = 0)(implicit sc: DistributedContext): CheckpointedDrm[_] = {

    val rdd = sc.sequenceFile(path, classOf[Writable], classOf[VectorWritable], minSplits = parMin)
        // Get rid of VectorWritable
        .map(t => (t._1, t._2.get()))

    def getKeyClassTag[K: ClassTag, V](rdd: RDD[(K, V)]) = implicitly[ClassTag[K]]

    // Spark should've loaded the type info from the header, right?
    val keyTag = getKeyClassTag(rdd)

    val (key2valFunc, val2keyFunc, unwrappedKeyTag) = keyTag match {

      case xx: ClassTag[Writable] if (xx == implicitly[ClassTag[IntWritable]]) => (
          (v: AnyRef) => v.asInstanceOf[IntWritable].get,
          (x: Any) => new IntWritable(x.asInstanceOf[Int]),
          implicitly[ClassTag[Int]])

      case xx: ClassTag[Writable] if (xx == implicitly[ClassTag[Text]]) => (
          (v: AnyRef) => v.asInstanceOf[Text].toString,
          (x: Any) => new Text(x.toString),
          implicitly[ClassTag[String]])

      case xx: ClassTag[Writable] if (xx == implicitly[ClassTag[LongWritable]]) => (
          (v: AnyRef) => v.asInstanceOf[LongWritable].get,
          (x: Any) => new LongWritable(x.asInstanceOf[Int]),
          implicitly[ClassTag[Long]])

      case xx: ClassTag[Writable] => (
          (v: AnyRef) => v,
          (x: Any) => x.asInstanceOf[Writable],
          ClassTag(classOf[Writable]))
    }

    {
      implicit def getWritable(x: Any): Writable = val2keyFunc()
      new CheckpointedDrmSpark(
        rdd = rdd.map(t => (key2valFunc(t._1), t._2)),
        _cacheStorageLevel = StorageLevel.MEMORY_ONLY
      )(unwrappedKeyTag.asInstanceOf[ClassTag[Any]])
    }
  }

  /** Parallelize in-core matrix as spark distributed matrix, using row ordinal indices as data set keys. */
  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)
      (implicit sc: DistributedContext)
  : CheckpointedDrm[Int] = {
    new CheckpointedDrmSpark(rdd = parallelizeInCore(m, numPartitions))
  }

  private[sparkbindings] def parallelizeInCore(m: Matrix, numPartitions: Int = 1)
      (implicit sc: DistributedContext): DrmRdd[Int] = {

    val p = (0 until m.nrow).map(i => i -> m(i, ::))
    sc.parallelize(p, numPartitions)

  }

  /** Parallelize in-core matrix as spark distributed matrix, using row labels as a data set keys. */
  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int = 1)
      (implicit sc: DistributedContext)
  : CheckpointedDrm[String] = {

    val rb = m.getRowLabelBindings
    val p = for (i: String <- rb.keySet().toIndexedSeq) yield i -> m(rb(i), ::)

    new CheckpointedDrmSpark(rdd = sc.parallelize(p, numPartitions))
  }

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int = 10)
      (implicit sc: DistributedContext): CheckpointedDrm[Int] = {
    val rdd = sc.parallelize(0 to numPartitions, numPartitions).flatMap(part => {
      val partNRow = (nrow - 1) / numPartitions + 1
      val partStart = partNRow * part
      val partEnd = Math.min(partStart + partNRow, nrow)

      for (i <- partStart until partEnd) yield (i, new RandomAccessSparseVector(ncol): Vector)
    })
    new CheckpointedDrmSpark[Int](rdd, nrow, ncol)
  }

  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int = 10)
      (implicit sc: DistributedContext): CheckpointedDrm[Long] = {
    val rdd = sc.parallelize(0 to numPartitions, numPartitions).flatMap(part => {
      val partNRow = (nrow - 1) / numPartitions + 1
      val partStart = partNRow * part
      val partEnd = Math.min(partStart + partNRow, nrow)

      for (i <- partStart until partEnd) yield (i, new RandomAccessSparseVector(ncol): Vector)
    })
    new CheckpointedDrmSpark[Long](rdd, nrow, ncol)
  }

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

  /** Translate previously optimized physical plan */
  private def tr2phys[K: ClassTag](oper: DrmLike[K]): DrmRddInput[K] = {
    // I do explicit evidence propagation here since matching via case classes seems to be loosing
    // it and subsequently may cause something like DrmRddInput[Any] instead of [Int] or [String].
    // Hence you see explicit evidence attached to all recursive exec() calls.
    oper match {
      // If there are any such cases, they must go away in pass1. If they were not, then it wasn't
      // the A'A case but actual transposition intent which should be removed from consideration
      // (we cannot do actual flip for non-int-keyed arguments)
      case OpAtAnyKey(_) =>
        throw new IllegalArgumentException("\"A\" must be Int-keyed in this A.t expression.")
      case op@OpAt(a) => At.at(op, tr2phys(a)(op.classTagA))
      case op@OpABt(a, b) => ABt.abt(op, tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpAtB(a, b) => AtB.atb_nograph(op, tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB),
        zippable = a.partitioningTag == b.partitioningTag)
      case op@OpAtA(a) => AtA.at_a(op, tr2phys(a)(op.classTagA))
      case op@OpAx(a, x) => Ax.ax_with_broadcast(op, tr2phys(a)(op.classTagA))
      case op@OpAtx(a, x) => Ax.atx_with_broadcast(op, tr2phys(a)(op.classTagA))
      case op@OpAewB(a, b, opId) => AewB.a_ew_b(op, tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpAewScalar(a, s, _) => AewB.a_ew_scalar(op, tr2phys(a)(op.classTagA), s)
      case op@OpRowRange(a, _) => Slicing.rowRange(op, tr2phys(a)(op.classTagA))
      case op@OpTimesRightMatrix(a, _) => AinCoreB.rightMultiply(op, tr2phys(a)(op.classTagA))
      // Custom operators, we just execute them
      case blockOp: OpMapBlock[K, _] => MapBlock.exec(
        src = tr2phys(blockOp.A)(blockOp.classTagA),
        ncol = blockOp.ncol,
        bmf = blockOp.bmf
      )
      case op@OpPar(a,_,_) => Par.exec(op,tr2phys(a)(op.classTagA))
      case cp: CheckpointedDrm[K] => new DrmRddInput[K](rowWiseSrc = Some((cp.ncol, cp.rdd)))
      case _ => throw new IllegalArgumentException("Internal:Optimizer has no exec policy for operator %s."
          .format(oper))

    }
  }


}

