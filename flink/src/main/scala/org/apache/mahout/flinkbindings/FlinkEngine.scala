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

import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.java.typeutils.TypeExtractor
import org.apache.flink.api.scala._
import org.apache.flink.api.scala.utils.DataSetUtils
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.mahout.flinkbindings.blas._
import org.apache.mahout.flinkbindings.drm._
import org.apache.mahout.flinkbindings.io.{HDFSUtil, Hadoop2HDFSUtil}
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.math.indexeddataset.{BiDictionary, IndexedDataset, Schema}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._

import scala.collection.JavaConversions._
import scala.reflect._

object FlinkEngine extends DistributedEngine {

  // By default, use Hadoop 2 utils
  var hdfsUtils: HDFSUtil = Hadoop2HDFSUtil

  /**
    * Load DRM from hdfs (as in Mahout DRM format).
    *
    * @param path   The DFS path to load from
    * @param parMin Minimum parallelism after load (equivalent to #par(min=...)).
    */
  override def drmDfsRead(path: String, parMin: Int = 1)
                         (implicit dc: DistributedContext): CheckpointedDrm[_] = {

    // Require that context is actually Flink context.
    require(dc.isInstanceOf[FlinkDistributedContext], "Supplied context must be for the Flink backend.")

    // Extract the Flink Environment variable
    implicit val env = dc.asInstanceOf[FlinkDistributedContext].env

    // set the parallelism of the env to parMin
    env.setParallelism(parMin)

    // get the header of a SequenceFile in the path
    val metadata = hdfsUtils.readDrmHeader(path + "//")

    val keyClass: Class[_] = metadata.keyTypeWritable

    // from the header determine which function to use to unwrap the key
    val unwrapKey = metadata.unwrapKeyFunction

    // Map to the correct DrmLike based on the metadata information
    if (metadata.keyClassTag == ClassTag.Int) {
      val ds = env.readSequenceFile(classOf[IntWritable], classOf[VectorWritable], path)

      val res = ds.map(new MapFunction[(IntWritable, VectorWritable), (Int, Vector)] {
        def map(tuple: (IntWritable, VectorWritable)): (Int, Vector) = {
          (unwrapKey(tuple._1).asInstanceOf[Int], tuple._2.get())
        }
      })
      datasetWrap(res)(metadata.keyClassTag.asInstanceOf[ClassTag[Int]])
    } else if (metadata.keyClassTag == ClassTag.Long) {
      val ds = env.readSequenceFile(classOf[LongWritable], classOf[VectorWritable], path)

      val res = ds.map(new MapFunction[(LongWritable, VectorWritable), (Long, Vector)] {
        def map(tuple: (LongWritable, VectorWritable)): (Long, Vector) = {
          (unwrapKey(tuple._1).asInstanceOf[Long], tuple._2.get())
        }
      })
      datasetWrap(res)(metadata.keyClassTag.asInstanceOf[ClassTag[Long]])
    } else if (metadata.keyClassTag == ClassTag(classOf[String])) {
      val ds = env.readSequenceFile(classOf[Text], classOf[VectorWritable], path)

      val res = ds.map(new MapFunction[(Text, VectorWritable), (String, Vector)] {
        def map(tuple: (Text, VectorWritable)): (String, Vector) = {
          (unwrapKey(tuple._1).asInstanceOf[String], tuple._2.get())
        }
      })
      datasetWrap(res)(metadata.keyClassTag.asInstanceOf[ClassTag[String]])
    } else throw new IllegalArgumentException(s"Unsupported DRM key type:${keyClass.getName}")

  }

  override def indexedDatasetDFSRead(src: String, schema: Schema, existingRowIDs: Option[BiDictionary])
                                    (implicit sc: DistributedContext): IndexedDataset = ???

  override def indexedDatasetDFSReadElements(src: String, schema: Schema, existingRowIDs: Option[BiDictionary])
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


  private def flinkTranslate[K](oper: DrmLike[K]): FlinkDrm[K] = {
    implicit val kTag = oper.keyClassTag
    implicit val typeInformation = generateTypeInformation[K]
    oper match {
      case OpAtAnyKey(_) ⇒
        throw new IllegalArgumentException("\"A\" must be Int-keyed in this A.t expression.")
      case op@OpAx(a, x) ⇒
        FlinkOpAx.blockifiedBroadcastAx(op, flinkTranslate(a))
      case op@OpAt(a) if op.keyClassTag == ClassTag.Int ⇒ FlinkOpAt.sparseTrick(op, flinkTranslate(a)).asInstanceOf[FlinkDrm[K]]
      case op@OpAtx(a, x) if op.keyClassTag == ClassTag.Int ⇒
        FlinkOpAx.atx_with_broadcast(op, flinkTranslate(a)).asInstanceOf[FlinkDrm[K]]
      case op@OpAtB(a, b) ⇒ FlinkOpAtB.notZippable(op, flinkTranslate(a),
        flinkTranslate(b)).asInstanceOf[FlinkDrm[K]]
      case op@OpABt(a, b) ⇒
        // express ABt via AtB: let C=At and D=Bt, and calculate CtD
        // TODO: create specific implementation of ABt, see MAHOUT-1750
        val opAt = OpAt(a.asInstanceOf[DrmLike[Int]]) // TODO: casts!
        val at = FlinkOpAt.sparseTrick(opAt, flinkTranslate(a.asInstanceOf[DrmLike[Int]]))
        val c = new CheckpointedFlinkDrm(at.asRowWise.ds, _nrow = opAt.nrow, _ncol = opAt.ncol)
        val opBt = OpAt(b.asInstanceOf[DrmLike[Int]]) // TODO: casts!
        val bt = FlinkOpAt.sparseTrick(opBt, flinkTranslate(b.asInstanceOf[DrmLike[Int]]))
        val d = new CheckpointedFlinkDrm(bt.asRowWise.ds, _nrow = opBt.nrow, _ncol = opBt.ncol)
        FlinkOpAtB.notZippable(OpAtB(c, d), flinkTranslate(c), flinkTranslate(d)).asInstanceOf[FlinkDrm[K]]
      case op@OpAtA(a) if op.keyClassTag == ClassTag.Int ⇒ FlinkOpAtA.at_a(op, flinkTranslate(a)).asInstanceOf[FlinkDrm[K]]
      case op@OpTimesRightMatrix(a, b) ⇒
        FlinkOpTimesRightMatrix.drmTimesInCore(op, flinkTranslate(a), b)
      case op@OpAewUnaryFunc(a, _, _) ⇒
        FlinkOpAewScalar.opUnaryFunction(op, flinkTranslate(a))
      case op@OpAewUnaryFuncFusion(a, _) ⇒
        FlinkOpAewScalar.opUnaryFunction(op, flinkTranslate(a))
      // deprecated
      case op@OpAewScalar(a, scalar, _) ⇒
        FlinkOpAewScalar.opScalarNoSideEffect(op, flinkTranslate(a), scalar)
      case op@OpAewB(a, b, _) ⇒
        FlinkOpAewB.rowWiseJoinNoSideEffect(op, flinkTranslate(a), flinkTranslate(b))
      case op@OpCbind(a, b) ⇒
        FlinkOpCBind.cbind(op, flinkTranslate(a), flinkTranslate(b))
      case op@OpRbind(a, b) ⇒
        FlinkOpRBind.rbind(op, flinkTranslate(a), flinkTranslate(b))
      case op@OpCbindScalar(a, x, _) ⇒
        FlinkOpCBind.cbindScalar(op, flinkTranslate(a), x)
      case op@OpRowRange(a, _) ⇒
        FlinkOpRowRange.slice(op, flinkTranslate(a)).asInstanceOf[FlinkDrm[K]]
      case op@OpABAnyKey(a, b) if a.keyClassTag != b.keyClassTag ⇒
        throw new IllegalArgumentException("DRMs A and B have different indices, cannot multiply them")
      case op: OpMapBlock[_, K] ⇒
        FlinkOpMapBlock.apply(flinkTranslate(op.A), op.ncol, op)
      case cp: CheckpointedDrm[K] ⇒ cp
      case _ ⇒
        throw new NotImplementedError(s"operator $oper is not implemented yet")
    }
  }

  /**
    * returns a vector that contains a column-wise sum from DRM
    */
  override def colSums[K](drm: CheckpointedDrm[K]): Vector = {
    implicit val kTag: ClassTag[K] = drm.keyClassTag
    implicit val typeInformation = generateTypeInformation[K]


    val sum = drm.ds.map {
      tuple => tuple._2
    }.reduce(_ + _)

    val list = sum.collect
    list.head
  }

  /** Engine-specific numNonZeroElementsPerColumn implementation based on a checkpoint. */
  override def numNonZeroElementsPerColumn[K](drm: CheckpointedDrm[K]): Vector = {
    implicit val kTag: ClassTag[K] = drm.keyClassTag
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
  override def colMeans[K](drm: CheckpointedDrm[K]): Vector = {
    drm.colSums() / drm.nrow
  }

  /**
    * Calculates the element-wise squared norm of a matrix
    */
  override def norm[K](drm: CheckpointedDrm[K]): Double = {
    implicit val kTag: ClassTag[K] = drm.keyClassTag
    implicit val typeInformation = generateTypeInformation[K]

    val sumOfSquares = drm.ds.map {
      tuple => tuple match {
        case (idx, vec) => vec dot vec
      }
    }.reduce(_ + _)

    val list = sumOfSquares.collect

    // check on this --why is it returning a list?
    math.sqrt(list.head)
  }

  /** Broadcast support */
  override def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] =
    FlinkByteBCast.wrap(v)

  /** Broadcast support */
  override def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] =
    FlinkByteBCast.wrap(m)

  /** Parallelize in-core matrix as flink distributed matrix, using row ordinal indices as data set keys. */
  // The 'numPartitions' parameter is not honored in this call,
  // as Flink sets a global parallelism in ExecutionEnvironment
  override def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)
                                           (implicit dc: DistributedContext): CheckpointedDrm[Int] = {

    val parallelDrm = parallelize(m, numPartitions)

    new CheckpointedFlinkDrm(ds = parallelDrm, _nrow = m.numRows(), _ncol = m.numCols())
  }

  // The 'parallelismDegree' parameter is not honored in this call,
  // as Flink sets a global parallelism in ExecutionEnvironment
  private[flinkbindings] def parallelize(m: Matrix, parallelismDegree: Int)
                                        (implicit dc: DistributedContext): DrmDataSet[Int] = {
    val rows = (0 until m.nrow).map(i => (i, m(i, ::)))
    val dataSetType = TypeExtractor.getForObject(rows.head)
    dc.env.fromCollection(rows).partitionByRange(0)
  }

  /** Parallelize in-core matrix as flink distributed matrix, using row labels as a data set keys. */
  // The 'numPartitions' parameter is not honored in this call,
  // as Flink sets a global parallelism in ExecutionEnvironment
  override def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int = 1)
                                          (implicit dc: DistributedContext): CheckpointedDrm[String] = {

    val rb = m.getRowLabelBindings
    val p = for (i: String ← rb.keySet().toIndexedSeq) yield i → m(rb(i), ::)

    new CheckpointedFlinkDrm[String](dc.env.fromCollection(p),
      _nrow = m.nrow, _ncol = m.ncol, cacheHint = CacheHint.NONE)
  }

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  override def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int = 10)
                                  (implicit dc: DistributedContext): CheckpointedDrm[Int] = {
    val nonParallelResult = (0 to numPartitions).flatMap { part ⇒
      val partNRow = (nrow - 1) / numPartitions + 1
      val partStart = partNRow * part
      val partEnd = Math.min(partStart + partNRow, nrow)

      for (i <- partStart until partEnd) yield (i, new RandomAccessSparseVector(ncol): Vector)
    }
    val result = dc.env.fromCollection(nonParallelResult)
    new CheckpointedFlinkDrm[Int](ds = result, _nrow = nrow, _ncol = ncol)
  }

  /** Creates empty DRM with non-trivial height */
  override def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int = 10)
                                      (implicit dc: DistributedContext): CheckpointedDrm[Long] = {

    val nonParallelResult = (0 to numPartitions).flatMap { part ⇒
        val partNRow = (nrow - 1) / numPartitions + 1
        val partStart = partNRow * part
        val partEnd = Math.min(partStart + partNRow, nrow)

      for (i ← partStart until partEnd) yield (i, new RandomAccessSparseVector(ncol): Vector)
    }

    val result = dc.env.fromCollection(nonParallelResult)
    new CheckpointedFlinkDrm[Long](ds = result, _nrow = nrow, _ncol = ncol, cacheHint = CacheHint.NONE)
  }

  /**
   * Convert non-int-keyed matrix to an int-keyed, computing optionally mapping from old keys
   * to row indices in the new one. The mapping, if requested, is returned as a 1-column matrix.
   */
  def drm2IntKeyed[K](drmX: DrmLike[K], computeMap: Boolean = false): (DrmLike[Int], Option[DrmLike[K]]) = {
    implicit val ktag = drmX.keyClassTag
    implicit val kTypeInformation = generateTypeInformation[K]

    if (ktag == ClassTag.Int) {
      drmX.asInstanceOf[DrmLike[Int]] → None
    } else {
      val drmXcp = drmX.checkpoint(CacheHint.MEMORY_ONLY)
      val ncol = drmXcp.asInstanceOf[CheckpointedFlinkDrm[K]].ncol
      val nrow = drmXcp.asInstanceOf[CheckpointedFlinkDrm[K]].nrow

      // Compute sequential int key numbering.
      val (intDataset, keyMap) = blas.rekeySeqInts(drmDataSet = drmXcp, computeMap = computeMap)

      // Convert computed key mapping to a matrix.
      val mxKeyMap = keyMap.map { dataSet ⇒
        datasetWrap(dataSet.map {
          tuple: (K, Int) => {
            val ordinal = tuple._2
            val key = tuple._1
            key -> (dvec(ordinal): Vector)
          }
        })
      }

      intDataset -> mxKeyMap
    }
  }

  /**
   * (Optional) Sampling operation.
   */
  def drmSampleRows[K](drmX: DrmLike[K], fraction: Double, replacement: Boolean = false): DrmLike[K] = {
    implicit val kTag: ClassTag[K] =  drmX.keyClassTag
    implicit val typeInformation = generateTypeInformation[K]

    val sample = DataSetUtils(drmX.dataset).sample(replacement, fraction)

    val res = if (kTag != ClassTag.Int) {
      new CheckpointedFlinkDrm[K](sample)
    }
    else {
      blas.rekeySeqInts(new RowsFlinkDrm[K](sample, ncol = drmX.ncol), computeMap = false)._1
        .asInstanceOf[DrmLike[K]]
    }

    res
  }

  def drmSampleKRows[K](drmX: DrmLike[K], numSamples:Int, replacement: Boolean = false): Matrix = {
    implicit val kTag: ClassTag[K] =  drmX.keyClassTag
    implicit val typeInformation = generateTypeInformation[K]

    val sample = DataSetUtils(drmX.dataset).sampleWithSize(replacement, numSamples)
    val sampleArray = sample.collect().toArray
    val isSparse = sampleArray.exists { case (_, vec) ⇒ !vec.isDense }

    val vectors = sampleArray.map(_._2)
    val labels = sampleArray.view.zipWithIndex
      .map { case ((key, _), idx) ⇒ key.toString → (idx: Integer) }.toMap

    val mx: Matrix = if (isSparse) sparse(vectors: _*) else dense(vectors)
    mx.setRowLabelBindings(labels)

    mx
  }

  /** Engine-specific all reduce tensor operation. */
  def allreduceBlock[K](drm: CheckpointedDrm[K], bmf: BlockMapFunc2[K], rf: BlockReduceFunc): Matrix = {
    implicit val kTag: ClassTag[K] = drm.keyClassTag
    implicit val typeInformation = generateTypeInformation[K]

    val res = drm.asBlockified.ds.map(par => bmf(par)).reduce(rf)
    res.collect().head
  }

  def generateTypeInformation[K: ClassTag]: TypeInformation[K] = {
    implicit val ktag = classTag[K]

    generateTypeInformationFromTag(ktag)
  }

  private def generateTypeInformationFromTag[K](tag: ClassTag[K]): TypeInformation[K] = {
    if (tag.runtimeClass.equals(classOf[Int])) {
      createTypeInformation[Int].asInstanceOf[TypeInformation[K]]
    } else if (tag.runtimeClass.equals(classOf[Long])) {
      createTypeInformation[Long].asInstanceOf[TypeInformation[K]]
    } else if (tag.runtimeClass.equals(classOf[String])) {
      createTypeInformation[String].asInstanceOf[TypeInformation[K]]
    } else {
      throw new IllegalArgumentException(s"index type $tag is not supported")
    }
  }
}