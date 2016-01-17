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

import org.apache.flink.api.common.typeinfo.TypeInformation

import scala.collection.JavaConversions._
import scala.reflect.ClassTag
import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.java.typeutils.TypeExtractor
import org.apache.hadoop.io.Writable
import org.apache.mahout.flinkbindings.blas._
import org.apache.mahout.flinkbindings.drm._
import org.apache.mahout.flinkbindings.io.HDFSUtil
import org.apache.mahout.flinkbindings.io.Hadoop1HDFSUtil
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.math.indexeddataset.BiDictionary
import org.apache.mahout.math.indexeddataset.IndexedDataset
import org.apache.mahout.math.indexeddataset.Schema
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._

import org.apache.flink.api.scala._
import org.apache.flink.api.scala.utils.DataSetUtils


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

    // Require that context is actually Flink context.
    require(dc.isInstanceOf[FlinkDistributedContext], "Supplied context must be for the Flink backend.")

    // Extract the Flink Environment variable
    implicit val env = dc.asInstanceOf[FlinkDistributedContext].env

    val metadata = hdfsUtils.readDrmHeader(path)

    val unwrapKey = metadata.unwrapKeyFunction

    val ds = env.readSequenceFile(classOf[Writable], classOf[VectorWritable], path)

    val res = ds.map(new MapFunction[(Writable, VectorWritable), (Any, Vector)] {
      def map(tuple: (Writable, VectorWritable)): (Any, Vector) = {
        (unwrapKey(tuple._1), tuple._2)
      }
    })

    datasetWrap(res)(metadata.keyClassTag.asInstanceOf[ClassTag[Any]])
  }

  override def indexedDatasetDFSRead(src: String, schema: Schema, existingRowIDs: Option[BiDictionary])
                                    (implicit sc: DistributedContext): IndexedDataset = ???

  override def indexedDatasetDFSReadElements(src: String,schema: Schema, existingRowIDs: Option[BiDictionary])
                                            (implicit sc: DistributedContext): IndexedDataset = ???


  /**
    * Perform default expression rewrite. Return physical plan that we can pass to exec(). <P>
    *
    * A particular physical engine implementation may choose to either use or not use these rewrites
    * as a useful basic rewriting rule.<P>
    */
  override def optimizerRewrite[K: ClassTag](action: DrmLike[K]): DrmLike[K] = super.optimizerRewrite(action)

  /** 
   * Translates logical plan into Flink execution plan. 
   **/
  override def toPhysical[K: ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] = {
    // Flink-specific Physical Plan translation.
    implicit val typeInformation = generateTypeInformation[K]
    val drm = flinkTranslate(plan)
    val newcp = new CheckpointedFlinkDrm(ds = drm.asRowWise.ds, _nrow = plan.nrow, _ncol = plan.ncol)
    newcp.cache()
  }

  private def flinkTranslate[K: ClassTag](oper: DrmLike[K]): FlinkDrm[K] = {
    implicit val typeInformation = generateTypeInformation[K]
    oper match {
      case OpAtAnyKey(_) ⇒
        throw new IllegalArgumentException("\"A\" must be Int-keyed in this A.t expression.")
      case op@OpAx(a, x) =>
        implicit val typeInformation = generateTypeInformation[K]
        FlinkOpAx.blockifiedBroadcastAx(op, flinkTranslate(a)(op.classTagA))
      case op@OpAt(a) => FlinkOpAt.sparseTrick(op, flinkTranslate(a)(op.classTagA))
      case op@OpAtx(a, x) =>
        // express Atx as (A.t) %*% x
        // TODO: create specific implementation of Atx, see MAHOUT-1749
        val opAt = OpAt(a)
        val at = FlinkOpAt.sparseTrick(opAt, flinkTranslate(a)(op.classTagA))
        val atCast = new CheckpointedFlinkDrm(at.asRowWise.ds, _nrow = opAt.nrow, _ncol = opAt.ncol)
        val opAx = OpAx(atCast, x)
        FlinkOpAx.blockifiedBroadcastAx(opAx, flinkTranslate(atCast)(op.classTagA))
      case op@OpAtB(a, b) => FlinkOpAtB.notZippable(op, flinkTranslate(a)(op.classTagA),
        flinkTranslate(b)(op.classTagA))
      case op@OpABt(a, b) =>
        // express ABt via AtB: let C=At and D=Bt, and calculate CtD
        // TODO: create specific implementation of ABt, see MAHOUT-1750
        val opAt = OpAt(a.asInstanceOf[DrmLike[Int]]) // TODO: casts!
        val at = FlinkOpAt.sparseTrick(opAt, flinkTranslate(a.asInstanceOf[DrmLike[Int]]))
        val c = new CheckpointedFlinkDrm(at.asRowWise.ds, _nrow = opAt.nrow, _ncol = opAt.ncol)
        val opBt = OpAt(b.asInstanceOf[DrmLike[Int]]) // TODO: casts!
        val bt = FlinkOpAt.sparseTrick(opBt, flinkTranslate(b.asInstanceOf[DrmLike[Int]]))
        val d = new CheckpointedFlinkDrm(bt.asRowWise.ds, _nrow = opBt.nrow, _ncol = opBt.ncol)
        FlinkOpAtB.notZippable(OpAtB(c, d), flinkTranslate(c), flinkTranslate(d)).asInstanceOf[FlinkDrm[K]]
      case op@OpAtA(a) => FlinkOpAtA.at_a(op, flinkTranslate(a)(op.classTagA))
      case op@OpTimesRightMatrix(a, b) =>
        FlinkOpTimesRightMatrix.drmTimesInCore(op, flinkTranslate(a)(op.classTagA), b)
      case op@OpAewUnaryFunc(a, _, _) =>
        FlinkOpAewScalar.opUnaryFunction(op, flinkTranslate(a)(op.classTagA))
      case op@OpAewUnaryFuncFusion(a, _) =>
        FlinkOpAewScalar.opUnaryFunction(op, flinkTranslate(a)(op.classTagA))
      // deprecated
      case op@OpAewScalar(a, scalar, _) =>
        FlinkOpAewScalar.opScalarNoSideEffect(op, flinkTranslate(a)(op.classTagA), scalar)
      case op@OpAewB(a, b, _) =>
        FlinkOpAewB.rowWiseJoinNoSideEffect(op, flinkTranslate(a)(op.classTagA), flinkTranslate(b)(op.classTagA))
      case op@OpCbind(a, b) =>
        FlinkOpCBind.cbind(op, flinkTranslate(a)(op.classTagA), flinkTranslate(b)(op.classTagA))
      case op@OpRbind(a, b) =>
        FlinkOpRBind.rbind(op, flinkTranslate(a)(op.classTagA), flinkTranslate(b)(op.classTagA))
      case op@OpCbindScalar(a, x, _) =>
        FlinkOpCBind.cbindScalar(op, flinkTranslate(a)(op.classTagA), x)
      case op@OpRowRange(a, _) =>
        FlinkOpRowRange.slice(op, flinkTranslate(a)(op.classTagA))
      case op@OpABAnyKey(a, b) if extractRealClassTag(a) != extractRealClassTag(b) =>
        throw new IllegalArgumentException("DRMs A and B have different indices, cannot multiply them")
      case op: OpMapBlock[K, _] =>
        FlinkOpMapBlock.apply(flinkTranslate(op.A)(op.classTagA), op.ncol, op.bmf)
      case cp: CheckpointedFlinkDrm[K] =>
//        val ds2incore = cp.ds.collect()
//        val ds2 = cp.executionEnvironment
//          .fromCollection(ds2incore)
//          .partitionByRange(0)
//          .setParallelism(cp.executionEnvironment.getParallelism)
//          .rebalance()
        val ds2 = cp.ds.rebalance.map(x => x).rebalance()

        new RowsFlinkDrm(ds2, cp.ncol)
      case _ =>
        throw new NotImplementedError(s"operator $oper is not implemented yet")
    }
  }

  /** 
   * returns a vector that contains a column-wise sum from DRM 
   */
  override def colSums[K: ClassTag](drm: CheckpointedDrm[K]): Vector = {
    implicit val typeInformation = generateTypeInformation[K]

    val sum = drm.ds.map {
      tuple => tuple._2
    }.reduce(_ + _)

    val list = sum.collect
    list.head
  }

  /** Engine-specific numNonZeroElementsPerColumn implementation based on a checkpoint. */
  override def numNonZeroElementsPerColumn[K: ClassTag](drm: CheckpointedDrm[K]): Vector = {
    implicit val typeInformation = generateTypeInformation[K]

    val result = drm.asBlockified.ds.map {
      tuple =>
        val block = tuple._2
        val acc = block(0, ::).like()

        block.foreach { v =>
          v.nonZeroes().foreach { el => acc(el.index()) = acc(el.index()) + 1 }
        }

        acc
    }.reduce(_ + _)

    val list = result.collect
    list.head
  }

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
    implicit val typeInformation = generateTypeInformation[K]

    val sumOfSquares = drm.ds.map {
      tuple => tuple match {
        case (idx, vec) => vec dot vec
      }
    }.reduce(_ + _)

//    val sumOfSquares = drm.ds.map(new MapFunction[(K, Vector), Double] {
//      def map(tuple: (K, Vector)): Double = tuple match {
//        case (idx, vec) => vec dot vec
//      }
//    }).reduce(new ReduceFunction[Double] {
//      def reduce(v1: Double, v2: Double) = v1 + v2
//    })

    val list = sumOfSquares.collect
    list.head
  }

  /** Broadcast support */
  override def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] = 
    FlinkByteBCast.wrap(v)


  /** Broadcast support */
  override def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] = 
    FlinkByteBCast.wrap(m)


  /** Parallelize in-core matrix as flink distributed matrix, using row ordinal indices as data set keys. */
  override def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)
                                           (implicit dc: DistributedContext): CheckpointedDrm[Int] = {

    val parallelDrm = parallelize(m, numPartitions)

    new CheckpointedFlinkDrm(ds=parallelDrm, _nrow=m.numRows(), _ncol=m.numCols())
  }


  private[flinkbindings] def parallelize(m: Matrix, parallelismDegree: Int)
      (implicit dc: DistributedContext): DrmDataSet[Int] = {
    val rows = (0 until m.nrow).map(i => (i, m(i, ::)))//.toSeq.sortWith((ii, jj) => ii._1 < jj._1)
    val dataSetType = TypeExtractor.getForObject(rows.head)
    //TODO: Make Sure that this is the correct partitioning scheme
    dc.env.fromCollection(rows)
            .partitionByRange(0)
            .setParallelism(parallelismDegree)
            .rebalance()
  }

  /** Parallelize in-core matrix as spark distributed matrix, using row labels as a data set keys. */
  override def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int = 1)
                                          (implicit dc: DistributedContext): CheckpointedDrm[String] = {
    ???
  }

  /** This creates an empty DRM with specified numb er of partitions and cardinality. */
  override def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int = 10)
                                  (implicit dc: DistributedContext): CheckpointedDrm[Int] = {
    val nonParallelResult = (0 to numPartitions).flatMap { part => 
      val partNRow = (nrow - 1) / numPartitions + 1
      val partStart = partNRow * part
      val partEnd = Math.min(partStart + partNRow, nrow)

      for (i <- partStart until partEnd) yield (i, new RandomAccessSparseVector(ncol): Vector)
    }
    val result = dc.env.fromCollection(nonParallelResult)
    new CheckpointedFlinkDrm(ds=result, _nrow=nrow, _ncol=ncol)
  }

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
   * (Optional) Sampling operation.
   */
  def drmSampleRows[K: ClassTag](drmX: DrmLike[K], fraction: Double, replacement: Boolean = false): DrmLike[K] = {
    implicit val typeInformation = generateTypeInformation[K]
    val sample = DataSetUtils(drmX.dataset).sample(replacement, fraction)
    new CheckpointedFlinkDrm[K](sample)
  }

  def drmSampleKRows[K: ClassTag](drmX: DrmLike[K], numSamples:Int, replacement: Boolean = false): Matrix = {
    implicit val typeInformation = generateTypeInformation[K]
    val sample = DataSetUtils(drmX.dataset).sampleWithSize(replacement, numSamples)
    new CheckpointedFlinkDrm[K](sample)
  }

  /** Optional engine-specific all reduce tensor operation. */
  def allreduceBlock[K: ClassTag](drm: CheckpointedDrm[K], bmf: BlockMapFunc2[K], rf: BlockReduceFunc): Matrix = 
    throw new UnsupportedOperationException("the operation allreduceBlock is not yet supported on Flink")

  private def generateTypeInformation[K: ClassTag]: TypeInformation[K] = {
    val tag = implicitly[ClassTag[K]]

    generateTypeInformationFromTag(tag)
  }

  private def generateTypeInformationFromTag[K](tag: ClassTag[K]): TypeInformation[K] = {
    if (tag.runtimeClass.equals(classOf[Int])) {
      createTypeInformation[Int].asInstanceOf[TypeInformation[K]]
    } else if (tag.runtimeClass.equals(classOf[Long])) {
      createTypeInformation[Long].asInstanceOf[TypeInformation[K]]
    } else if (tag.runtimeClass.equals(classOf[String])) {
      createTypeInformation[String].asInstanceOf[TypeInformation[K]]
    } else if (tag.runtimeClass.equals(classOf[Any])) {
       createTypeInformation[Any].asInstanceOf[TypeInformation[K]]
    } else {
      throw new IllegalArgumentException(s"index type $tag is not supported")
    }
  }
}