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

import org.apache.hadoop.io._
import org.apache.mahout.common.{HDFSUtil, Hadoop1HDFSUtil}
import org.apache.mahout.drivers.TextDelimitedIndexedDatasetReader
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.math.indexeddataset.{BiDictionary, DefaultIndexedDatasetElementReadSchema, DefaultIndexedDatasetReadSchema, Schema}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.sparkbindings.blas._
import org.apache.mahout.sparkbindings.drm.{CheckpointedDrmSpark, DrmRddInput, cpDrmGeneric2DrmRddInput}
import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark
import org.apache.spark.storage.StorageLevel

import scala.collection.JavaConversions._
import scala.collection._
import scala.reflect.ClassTag

/** Spark-specific non-drm-method operations */
object SparkEngine extends DistributedEngine {

  // By default, use Hadoop 1 utils
  var hdfsUtils: HDFSUtil = Hadoop1HDFSUtil

  def colSums[K](drm: CheckpointedDrm[K]): Vector = {
    val n = drm.ncol

    drm.rdd

      // Throw away keys
      .map(_._2)

      // Fold() doesn't work with kryo still. So work around it.
      .mapPartitions(iter ⇒ {
      val acc = ((new DenseVector(n): Vector) /: iter) ((acc, v) ⇒ acc += v)
      Iterator(acc)
    })

      // Since we preallocated new accumulator vector per partition, this must not cause any side
      // effects now.
      .reduce(_ += _)
  }

  def numNonZeroElementsPerColumn[K](drm: CheckpointedDrm[K]): Vector = {
    val n = drm.ncol

    drm.rdd

      // Throw away keys
      .map(_._2)

      // Fold() doesn't work with kryo still. So work around it.
      .mapPartitions(iter ⇒ {
      val acc = ((new DenseVector(n): Vector) /: iter) { (acc, v) ⇒
        v.nonZeroes().foreach { elem ⇒ acc(elem.index) += 1 }
        acc
      }
      Iterator(acc)
    })
      // Since we preallocated new accumulator vector per partition, this must not cause any side
      // effects now.
      .reduce(_ += _)
  }

  /** Engine-specific colMeans implementation based on a checkpoint. */
  override def colMeans[K](drm: CheckpointedDrm[K]): Vector =
    if (drm.nrow == 0) drm.colSums() else drm.colSums() /= drm.nrow

  override def norm[K](drm: CheckpointedDrm[K]): Double =
    math.sqrt(drm.rdd
      // Compute sum of squares of each vector
      .map {
      case (key, v) ⇒ v dot v
    }
      .reduce(_ + _))


  /** Optional engine-specific all reduce tensor operation. */
  override def allreduceBlock[K](drm: CheckpointedDrm[K], bmf: BlockMapFunc2[K], rf:
  BlockReduceFunc): Matrix = {
    drm.asBlockified(ncol = drm.ncol).map(bmf(_)).reduce(rf)
  }

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
    val rddInput = tr2phys(plan)

    val newcp = new CheckpointedDrmSpark(
      rddInput = rddInput,
      _nrow = plan.nrow,
      _ncol = plan.ncol,
      cacheHint = ch,
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
    * @return DRM[Any] where Any is automatically translated to value type
    */
  def drmDfsRead(path: String, parMin: Int = 0)(implicit sc: DistributedContext): CheckpointedDrm[_] = {

    // Require that context is actually Spark context.
    require(sc.isInstanceOf[SparkDistributedContext], "Supplied context must be for the Spark backend.")

    // Extract spark context -- we need it for some operations.
    implicit val ssc = sc.asInstanceOf[SparkDistributedContext].sc

    val drmMetadata = hdfsUtils.readDrmHeader(path)
    val k2vFunc = drmMetadata.keyW2ValFunc

    // Load RDD and convert all Writables to value types right away (due to reuse of writables in
    // Hadoop we must do it right after read operation).
    val rdd = sc.sequenceFile(path, classOf[Writable], classOf[VectorWritable], minPartitions = parMin)

      // Immediately convert keys and value writables into value types.
      .map { case (wKey, wVec) ⇒ k2vFunc(wKey) -> wVec.get() }

    // Wrap into a DRM type with correct matrix row key class tag evident.
    drmWrap(rdd = rdd, cacheHint = CacheHint.NONE)(drmMetadata.keyClassTag.asInstanceOf[ClassTag[Any]])
  }

  /** Parallelize in-core matrix as spark distributed matrix, using row ordinal indices as data set keys. */
  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)
                                  (implicit sc: DistributedContext)
  : CheckpointedDrm[Int] = {
    new CheckpointedDrmSpark(rddInput = parallelizeInCore(m, numPartitions), _nrow = m.nrow, _ncol = m.ncol,
      cacheHint = CacheHint.NONE)
  }

  private[sparkbindings] def parallelizeInCore(m: Matrix, numPartitions: Int = 1)
                                              (implicit sc: DistributedContext): DrmRdd[Int] = {

    val p = (0 until m.nrow).map(i ⇒ i → m(i, ::))
    sc.parallelize(p, numPartitions)

  }

  /** Parallelize in-core matrix as spark distributed matrix, using row labels as a data set keys. */
  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int = 1)
                                 (implicit sc: DistributedContext)
  : CheckpointedDrm[String] = {

    val rb = m.getRowLabelBindings
    val p = for (i: String ← rb.keySet().toIndexedSeq) yield i → m(rb(i), ::)

    new CheckpointedDrmSpark(rddInput = sc.parallelize(p, numPartitions), _nrow = m.nrow, _ncol = m.ncol,
      cacheHint = CacheHint.NONE)
  }

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int = 10)
                         (implicit sc: DistributedContext): CheckpointedDrm[Int] = {
    val rdd = sc.parallelize(0 to numPartitions, numPartitions).flatMap(part ⇒ {
      val partNRow = (nrow - 1) / numPartitions + 1
      val partStart = partNRow * part
      val partEnd = Math.min(partStart + partNRow, nrow)

      for (i ← partStart until partEnd) yield (i, new RandomAccessSparseVector(ncol): Vector)
    })
    new CheckpointedDrmSpark[Int](rdd, nrow, ncol, cacheHint = CacheHint.NONE)
  }

  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int = 10)
                             (implicit sc: DistributedContext): CheckpointedDrm[Long] = {
    val rdd = sc.parallelize(0 to numPartitions, numPartitions).flatMap(part ⇒ {
      val partNRow = (nrow - 1) / numPartitions + 1
      val partStart = partNRow * part
      val partEnd = Math.min(partStart + partNRow, nrow)

      for (i ← partStart until partEnd) yield (i, new RandomAccessSparseVector(ncol): Vector)
    })
    new CheckpointedDrmSpark[Long](rdd, nrow, ncol, cacheHint = CacheHint.NONE)
  }

  /**
    * Convert non-int-keyed matrix to an int-keyed, computing optionally mapping from old keys
    * to row indices in the new one. The mapping, if requested, is returned as a 1-column matrix.
    */
  override def drm2IntKeyed[K](drmX: DrmLike[K], computeMap: Boolean = false): (DrmLike[Int], Option[DrmLike[K]]) = {
    implicit val ktag = drmX.keyClassTag
    if (ktag == ClassTag.Int) {
      drmX.asInstanceOf[DrmLike[Int]] → None
    } else {

      val drmXcp = drmX.checkpoint(CacheHint.MEMORY_ONLY)
      val ncol = drmXcp.asInstanceOf[CheckpointedDrmSpark[K]]._ncol
      val nrow = drmXcp.asInstanceOf[CheckpointedDrmSpark[K]]._nrow

      // Compute sequential int key numbering.
      val (intRdd, keyMap) = blas.rekeySeqInts(rdd = drmXcp.rdd, computeMap = computeMap)

      // Convert computed key mapping to a matrix.
      val mxKeyMap = keyMap.map { rdd ⇒
        drmWrap(rdd = rdd.map { case (key, ordinal) ⇒ key → (dvec(ordinal): Vector) }, ncol = 1, nrow = nrow)
      }


      drmWrap(rdd = intRdd, ncol = ncol) → mxKeyMap
    }

  }


  /**
    * (Optional) Sampling operation. Consistent with Spark semantics of the same.
    *
    * @param drmX
    * @param fraction
    * @param replacement
    * @tparam K
    * @return
    */
  override def drmSampleRows[K](drmX: DrmLike[K], fraction: Double, replacement: Boolean): DrmLike[K] = {

    implicit val ktag = drmX.keyClassTag

    // We do want to take ncol if already computed, if not, then we don't want to trigger computation
    // here.
    val ncol = drmX match {
      case cp: CheckpointedDrmSpark[K] ⇒ cp._ncol
      case _ ⇒ -1
    }
    val sample = drmX.rdd.sample(withReplacement = replacement, fraction = fraction)
    if (ktag != ClassTag.Int) return drmWrap(sample, ncol = ncol)

    // K == Int: Int-keyed sample. rebase int counts.
    drmWrap(rdd = blas.rekeySeqInts(rdd = sample, computeMap = false)._1, ncol = ncol).asInstanceOf[DrmLike[K]]
  }


  override def drmSampleKRows[K](drmX: DrmLike[K], numSamples: Int, replacement: Boolean): Matrix = {

    val ncol = drmX match {
      case cp: CheckpointedDrmSpark[K] ⇒ cp._ncol
      case _ ⇒ -1
    }

    // I think as of the time of this writing, takeSample() in Spark is biased. It is not a true
    // hypergeometric sampler. But it is faster than a true hypergeometric/categorical samplers
    // would be.
    val sample = drmX.rdd.takeSample(withReplacement = replacement, num = numSamples)
    val isSparse = sample.exists { case (_, vec) ⇒ !vec.isDense }

    val vectors = sample.map(_._2)
    val labels = sample.view.zipWithIndex.map { case ((key, _), idx) ⇒ key.toString → (idx: Integer) }.toMap

    val mx: Matrix = if (isSparse) sparse(vectors: _*) else dense(vectors)
    mx.setRowLabelBindings(labels)

    mx
  }

  private[mahout] def cacheHint2Spark(cacheHint: CacheHint.CacheHint): StorageLevel = cacheHint match {
    case CacheHint.NONE ⇒ StorageLevel.NONE
    case CacheHint.DISK_ONLY ⇒ StorageLevel.DISK_ONLY
    case CacheHint.DISK_ONLY_2 ⇒ StorageLevel.DISK_ONLY_2
    case CacheHint.MEMORY_ONLY ⇒ StorageLevel.MEMORY_ONLY
    case CacheHint.MEMORY_ONLY_2 ⇒ StorageLevel.MEMORY_ONLY_2
    case CacheHint.MEMORY_ONLY_SER ⇒ StorageLevel.MEMORY_ONLY_SER
    case CacheHint.MEMORY_ONLY_SER_2 ⇒ StorageLevel.MEMORY_ONLY_SER_2
    case CacheHint.MEMORY_AND_DISK ⇒ StorageLevel.MEMORY_AND_DISK
    case CacheHint.MEMORY_AND_DISK_2 ⇒ StorageLevel.MEMORY_AND_DISK_2
    case CacheHint.MEMORY_AND_DISK_SER ⇒ StorageLevel.MEMORY_AND_DISK_SER
    case CacheHint.MEMORY_AND_DISK_SER_2 ⇒ StorageLevel.MEMORY_AND_DISK_SER_2
  }

  /** Translate previously optimized physical plan */
  private def tr2phys[K](oper: DrmLike[K]): DrmRddInput[K] = {
    // I do explicit evidence propagation here since matching via case classes seems to be loosing
    // it and subsequently may cause something like DrmRddInput[Any] instead of [Int] or [String].
    // Hence you see explicit evidence attached to all recursive exec() calls.
    oper match {
      // If there are any such cases, they must go away in pass1. If they were not, then it wasn't
      // the A'A case but actual transposition intent which should be removed from consideration
      // (we cannot do actual flip for non-int-keyed arguments)
      case OpAtAnyKey(_) ⇒
        throw new IllegalArgumentException("\"A\" must be Int-keyed in this A.t expression.")
      case op@OpAt(a) if op.keyClassTag == ClassTag.Int ⇒ At.at(op, tr2phys(a)).asInstanceOf[DrmRddInput[K]]
      case op@OpABt(a, b) ⇒ ABt.abt(op, tr2phys(a), tr2phys(b))
      case op@OpAtB(a, b) ⇒ AtB.atb(op, tr2phys(a), tr2phys(b)).asInstanceOf[DrmRddInput[K]]
      case op@OpAtA(a) if op.keyClassTag == ClassTag.Int ⇒ AtA.at_a(op, tr2phys(a)).asInstanceOf[DrmRddInput[K]]
      case op@OpAx(a, x) ⇒ Ax.ax_with_broadcast(op, tr2phys(a))
      case op@OpAtx(a, x) if op.keyClassTag == ClassTag.Int ⇒
        Ax.atx_with_broadcast(op, tr2phys(a)).asInstanceOf[DrmRddInput[K]]
      case op@OpAewUnaryFunc(a, _, _) ⇒ AewB.a_ew_func(op, tr2phys(a))
      case op@OpAewUnaryFuncFusion(a, _) ⇒ AewB.a_ew_func(op, tr2phys(a))
      case op@OpAewB(a, b, opId) ⇒ AewB.a_ew_b(op, tr2phys(a), tr2phys(b))
      case op@OpCbind(a, b) ⇒ CbindAB.cbindAB_nograph(op, tr2phys(a), tr2phys(b))
      case op@OpCbindScalar(a, _, _) ⇒ CbindAB.cbindAScalar(op, tr2phys(a))
      case op@OpRbind(a, b) ⇒ RbindAB.rbindAB(op, tr2phys(a), tr2phys(b))
      case op@OpAewScalar(a, s, _) ⇒ AewB.a_ew_scalar(op, tr2phys(a), s)
      case op@OpRowRange(a, _) if op.keyClassTag == ClassTag.Int ⇒
        Slicing.rowRange(op, tr2phys(a)).asInstanceOf[DrmRddInput[K]]
      case op@OpTimesRightMatrix(a, _) ⇒ AinCoreB.rightMultiply(op, tr2phys(a))
      // Custom operators, we just execute them
      case blockOp: OpMapBlock[_, K] ⇒ MapBlock.exec(
        src = tr2phys(blockOp.A),
        operator = blockOp
      )
      case op@OpPar(a, _, _) ⇒ Par.exec(op, tr2phys(a))
      case cp: CheckpointedDrm[K] ⇒
        implicit val ktag=cp.keyClassTag
        cp.rdd: DrmRddInput[K]
      case _ ⇒ throw new IllegalArgumentException("Internal:Optimizer has no exec policy for operator %s."
        .format(oper))

    }
  }

  /**
    * Returns an [[org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark]] from default text
    * delimited files. Reads a vector per row.
    *
    * @param src    a comma separated list of URIs to read from
    * @param schema how the text file is formatted
    */
  def indexedDatasetDFSRead(src: String,
                            schema: Schema = DefaultIndexedDatasetReadSchema,
                            existingRowIDs: Option[BiDictionary] = None)
                           (implicit sc: DistributedContext):
  IndexedDatasetSpark = {
    val reader = new TextDelimitedIndexedDatasetReader(schema)(sc)
    val ids = reader.readRowsFrom(src, existingRowIDs)
    ids
  }

  /**
    * Returns an [[org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark]] from default text
    * delimited files. Reads an element per row.
    *
    * @param src    a comma separated list of URIs to read from
    * @param schema how the text file is formatted
    */
  def indexedDatasetDFSReadElements(src: String,
                                    schema: Schema = DefaultIndexedDatasetElementReadSchema,
                                    existingRowIDs: Option[BiDictionary] = None)
                                   (implicit sc: DistributedContext):
  IndexedDatasetSpark = {
    val reader = new TextDelimitedIndexedDatasetReader(schema)(sc)
    val ids = reader.readElementsFrom(src, existingRowIDs)
    ids
  }

}

