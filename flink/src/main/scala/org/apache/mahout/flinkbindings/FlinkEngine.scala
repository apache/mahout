/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.mahout.flinkbindings

import java.util.Collection

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.common.functions.ReduceFunction
import org.apache.flink.api.java.tuple.Tuple2
import org.apache.flink.api.java.typeutils.TypeExtractor
import org.apache.hadoop.io.Writable
import org.apache.hadoop.mapred.FileInputFormat
import org.apache.hadoop.mapred.JobConf
import org.apache.hadoop.mapred.SequenceFileInputFormat
import org.apache.mahout.flinkbindings.blas.FlinkOpAewB
import org.apache.mahout.flinkbindings.blas.FlinkOpAewScalar
import org.apache.mahout.flinkbindings.blas.FlinkOpAt
import org.apache.mahout.flinkbindings.blas.FlinkOpAtB
import org.apache.mahout.flinkbindings.blas.FlinkOpAx
import org.apache.mahout.flinkbindings.blas.FlinkOpCBind
import org.apache.mahout.flinkbindings.blas.FlinkOpMapBlock
import org.apache.mahout.flinkbindings.blas.FlinkOpRBind
import org.apache.mahout.flinkbindings.blas.FlinkOpRowRange
import org.apache.mahout.flinkbindings.blas.FlinkOpTimesRightMatrix
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.CheckpointedFlinkDrm
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.flinkbindings.io.HDFSUtil
import org.apache.mahout.flinkbindings.io.Hadoop1HDFSUtil
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.Vector
import org.apache.mahout.math.VectorWritable
import org.apache.mahout.math.drm.BCast
import org.apache.mahout.math.drm.BlockMapFunc2
import org.apache.mahout.math.drm.BlockReduceFunc
import org.apache.mahout.math.drm.CacheHint
import org.apache.mahout.math.drm.CheckpointedDrm
import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.math.drm.DistributedEngine
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.DrmTuple
import org.apache.mahout.math.drm.drm2drmCpOps
import org.apache.mahout.math.drm.logical.OpABt
import org.apache.mahout.math.drm.logical.OpAewB
import org.apache.mahout.math.drm.logical.OpAewScalar
import org.apache.mahout.math.drm.logical.OpAewUnaryFunc
import org.apache.mahout.math.drm.logical.OpAt
import org.apache.mahout.math.drm.logical.OpAtA
import org.apache.mahout.math.drm.logical.OpAtB
import org.apache.mahout.math.drm.logical.OpAtx
import org.apache.mahout.math.drm.logical.OpAx
import org.apache.mahout.math.drm.logical.OpCbind
import org.apache.mahout.math.drm.logical.OpMapBlock
import org.apache.mahout.math.drm.logical.OpRbind
import org.apache.mahout.math.drm.logical.OpRowRange
import org.apache.mahout.math.drm.logical.OpTimesRightMatrix
import org.apache.mahout.math.indexeddataset.BiDictionary
import org.apache.mahout.math.indexeddataset.IndexedDataset
import org.apache.mahout.math.indexeddataset.Schema
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._

object FlinkEngine extends DistributedEngine {

  // By default, use Hadoop 1 utils
  var hdfsUtils: HDFSUtil = Hadoop1HDFSUtil

  /**
   * Load DRM from hdfs (as in Mahout DRM format).
   * 
   * @param path The DFS path to load from
   * @param parMin Minimum parallelism after load (equivalent to #par(min=...)).
   */
  override def drmDfsRead(path: String, parMin: Int = 0)
                         (implicit dc: DistributedContext): CheckpointedDrm[_] = {
    val metadata = hdfsUtils.readDrmHeader(path)
    val unwrapKey = metadata.unwrapKeyFunction

    val job = new JobConf
    val hadoopInput = new SequenceFileInputFormat[Writable, VectorWritable]
    FileInputFormat.addInputPath(job, new org.apache.hadoop.fs.Path(path))

    val writables = dc.env.createHadoopInput(hadoopInput, classOf[Writable], classOf[VectorWritable], job)

    val res = writables.map(new MapFunction[Tuple2[Writable, VectorWritable], (Any, Vector)] {
      def map(tuple: Tuple2[Writable, VectorWritable]): (Any, Vector) = {
        (unwrapKey(tuple.f0), tuple.f1)
      }
    })

    datasetWrap(res)(metadata.keyClassTag.asInstanceOf[ClassTag[Any]])
  }

  override def indexedDatasetDFSRead(src: String, schema: Schema, existingRowIDs: Option[BiDictionary])
                                    (implicit sc: DistributedContext): IndexedDataset = ???

  override def indexedDatasetDFSReadElements(src: String,schema: Schema, existingRowIDs: Option[BiDictionary])
                                            (implicit sc: DistributedContext): IndexedDataset = ???


  /** 
   * Translates logical plan into Flink execution plan. 
   **/
  override def toPhysical[K: ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] = {
    // Flink-specific Physical Plan translation.
    val drm = flinkTranslate(plan)

    // to Help Flink's type inference had to use just one specific type - Int 
    // see org.apache.mahout.flinkbindings.blas classes with TODO: casting inside
    // see MAHOUT-1747 and MAHOUT-1748
    val cls = implicitly[ClassTag[K]]
    if (!cls.runtimeClass.equals(classOf[Int])) {
      throw new IllegalArgumentException(s"At the moment only Int indexes are supported. Got $cls")
    }

    val newcp = new CheckpointedFlinkDrm(ds = drm.deblockify.ds, _nrow = plan.nrow, _ncol = plan.ncol)
    newcp.cache()
  }

  private def flinkTranslate[K: ClassTag](oper: DrmLike[K]): FlinkDrm[K] = oper match {
    case op @ OpAx(a, x) => FlinkOpAx.blockifiedBroadcastAx(op, flinkTranslate(a)(op.classTagA))
    case op @ OpAt(a) => FlinkOpAt.sparseTrick(op, flinkTranslate(a)(op.classTagA))
    case op @ OpAtx(a, x) => {
      // express Atx as (A.t) %*% x
      // TODO: create specific implementation of Atx, see MAHOUT-1749 
      val opAt = OpAt(a)
      val at = FlinkOpAt.sparseTrick(opAt, flinkTranslate(a)(op.classTagA))
      val atCast = new CheckpointedFlinkDrm(at.deblockify.ds, _nrow=opAt.nrow, _ncol=opAt.ncol)
      val opAx = OpAx(atCast, x)
      FlinkOpAx.blockifiedBroadcastAx(opAx, flinkTranslate(atCast)(op.classTagA))
    }
    case op @ OpAtB(a, b) => FlinkOpAtB.notZippable(op, flinkTranslate(a)(op.classTagA), 
        flinkTranslate(b)(op.classTagA))
    case op @ OpABt(a, b) => {
      // express ABt via AtB: let C=At and D=Bt, and calculate CtD
      // TODO: create specific implementation of ABt, see MAHOUT-1750 
      val opAt = OpAt(a.asInstanceOf[DrmLike[Int]]) // TODO: casts!
      val at = FlinkOpAt.sparseTrick(opAt, flinkTranslate(a.asInstanceOf[DrmLike[Int]]))
      val c = new CheckpointedFlinkDrm(at.deblockify.ds, _nrow=opAt.nrow, _ncol=opAt.ncol)

      val opBt = OpAt(b.asInstanceOf[DrmLike[Int]]) // TODO: casts!
      val bt = FlinkOpAt.sparseTrick(opBt, flinkTranslate(b.asInstanceOf[DrmLike[Int]]))
      val d = new CheckpointedFlinkDrm(bt.deblockify.ds, _nrow=opBt.nrow, _ncol=opBt.ncol)

      FlinkOpAtB.notZippable(OpAtB(c, d), flinkTranslate(c), flinkTranslate(d))
                .asInstanceOf[FlinkDrm[K]]
    }
    case op @ OpAtA(a) => {
      // express AtA via AtB
      // TODO: create specific implementation of AtA, see MAHOUT-1751 
      val aInt = a.asInstanceOf[DrmLike[Int]] // TODO: casts!
      val opAtB = OpAtB(aInt, aInt)
      val aTranslated = flinkTranslate(aInt)
      FlinkOpAtB.notZippable(opAtB, aTranslated, aTranslated)
    }
    case op @ OpTimesRightMatrix(a, b) => 
      FlinkOpTimesRightMatrix.drmTimesInCore(op, flinkTranslate(a)(op.classTagA), b)
    case op @ OpAewUnaryFunc(a, f, _) =>
      FlinkOpAewScalar.opUnaryFunction(op, flinkTranslate(a)(op.classTagA), f)
    case op @ OpAewScalar(a, scalar, _) => 
      FlinkOpAewScalar.opScalarNoSideEffect(op, flinkTranslate(a)(op.classTagA), scalar)
    case op @ OpAewB(a, b, _) =>
      FlinkOpAewB.rowWiseJoinNoSideEffect(op, flinkTranslate(a)(op.classTagA), flinkTranslate(b)(op.classTagA))
    case op @ OpCbind(a, b) => 
      FlinkOpCBind.cbind(op, flinkTranslate(a)(op.classTagA), flinkTranslate(b)(op.classTagA))
    case op @ OpRbind(a, b) => 
      FlinkOpRBind.rbind(op, flinkTranslate(a)(op.classTagA), flinkTranslate(b)(op.classTagA))
    case op @ OpRowRange(a, _) => 
      FlinkOpRowRange.slice(op, flinkTranslate(a)(op.classTagA))
    case op: OpMapBlock[K, _] => 
      FlinkOpMapBlock.apply(flinkTranslate(op.A)(op.classTagA), op.ncol, op.bmf)
    case cp: CheckpointedFlinkDrm[K] => new RowsFlinkDrm(cp.ds, cp.ncol)
    case _ => throw new NotImplementedError(s"operator $oper is not implemented yet")
  }

  /** 
   * returns a vector that contains a column-wise sum from DRM 
   */
  override def colSums[K: ClassTag](drm: CheckpointedDrm[K]): Vector = {
    val sum = drm.ds.map(new MapFunction[(K, Vector), Vector] {
      def map(tuple: (K, Vector)): Vector = tuple._2
    }).reduce(new ReduceFunction[Vector] {
      def reduce(v1: Vector, v2: Vector) = v1 + v2
    })

    val list = sum.collect.asScala.toList
    list.head
  }

  /** Engine-specific numNonZeroElementsPerColumn implementation based on a checkpoint. */
  override def numNonZeroElementsPerColumn[K: ClassTag](drm: CheckpointedDrm[K]): Vector = ???

  /** 
   * returns a vector that contains a column-wise mean from DRM 
   */
  override def colMeans[K: ClassTag](drm: CheckpointedDrm[K]): Vector = {
    drm.colSums() / drm.nrow
  }

  /**
   * Calculates the element-wise squared norm of a matrix
   */
  override def norm[K: ClassTag](drm: CheckpointedDrm[K]): Double = {
    val sumOfSquares = drm.ds.map(new MapFunction[(K, Vector), Double] {
      def map(tuple: (K, Vector)): Double = tuple match {
        case (idx, vec) => vec dot vec
      }
    }).reduce(new ReduceFunction[Double] {
      def reduce(v1: Double, v2: Double) = v1 + v2
    })

    val list = sumOfSquares.collect.asScala.toList
    list.head
  }

  /** Broadcast support */
  override def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] = 
    FlinkByteBCast.wrap(v)


  /** Broadcast support */
  override def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] = 
    FlinkByteBCast.wrap(m)


  /** Parallelize in-core matrix as spark distributed matrix, using row ordinal indices as data set keys. */
  override def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)
                                           (implicit dc: DistributedContext): CheckpointedDrm[Int] = {
    val parallelDrm = parallelize(m, numPartitions)
    new CheckpointedFlinkDrm(ds=parallelDrm, _nrow=m.numRows(), _ncol=m.numCols())
  }

  private[flinkbindings] def parallelize(m: Matrix, parallelismDegree: Int)
      (implicit dc: DistributedContext): DrmDataSet[Int] = {
    val rows = (0 until m.nrow).map(i => (i, m(i, ::)))
    val rowsJava: Collection[DrmTuple[Int]]  = rows.asJava

    val dataSetType = TypeExtractor.getForObject(rows.head)
    dc.env.fromCollection(rowsJava, dataSetType).setParallelism(parallelismDegree)
  }

  /** Parallelize in-core matrix as spark distributed matrix, using row labels as a data set keys. */
  override def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int = 1)
                                          (implicit sc: DistributedContext): CheckpointedDrm[String] = ???

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  override def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int = 10)
                                  (implicit sc: DistributedContext): CheckpointedDrm[Int] = ???

  /** Creates empty DRM with non-trivial height */
  override def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int = 10)
                                      (implicit sc: DistributedContext): CheckpointedDrm[Long] = ???
  

  /**
   * Convert non-int-keyed matrix to an int-keyed, computing optionally mapping from old keys
   * to row indices in the new one. The mapping, if requested, is returned as a 1-column matrix.
   */
  def drm2IntKeyed[K: ClassTag](drmX: DrmLike[K], computeMap: Boolean = false): 
          (DrmLike[Int], Option[DrmLike[K]]) = ???

  /**
   * (Optional) Sampling operation. Consistent with Spark semantics of the same.
   */
  def drmSampleRows[K: ClassTag](drmX: DrmLike[K], fraction: Double, replacement: Boolean = false): DrmLike[K] = ???

  def drmSampleKRows[K: ClassTag](drmX: DrmLike[K], numSamples:Int, replacement: Boolean = false): Matrix = ???

  /** Optional engine-specific all reduce tensor operation. */
  def allreduceBlock[K: ClassTag](drm: CheckpointedDrm[K], bmf: BlockMapFunc2[K], rf: BlockReduceFunc): Matrix = ???
 
}