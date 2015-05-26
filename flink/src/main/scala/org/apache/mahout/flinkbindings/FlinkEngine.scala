package org.apache.mahout.flinkbindings

import scala.reflect.ClassTag
import org.apache.flink.api.scala.DataSet
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.CheckpointedFlinkDrm
import org.apache.mahout.math._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm.BCast
import org.apache.mahout.math.drm.CacheHint
import org.apache.mahout.math.drm.CheckpointedDrm
import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.math.drm.DistributedEngine
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.indexeddataset.DefaultIndexedDatasetElementReadSchema
import org.apache.mahout.math.indexeddataset.DefaultIndexedDatasetReadSchema
import org.apache.mahout.math.indexeddataset.IndexedDataset
import org.apache.mahout.math.indexeddataset.Schema
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import com.google.common.collect.BiMap
import com.google.common.collect.HashBiMap
import scala.collection.JavaConverters._
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.java.typeutils.TypeExtractor
import org.apache.mahout.math.drm.DrmTuple
import java.util.Collection
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.blas._
import org.apache.mahout.math.drm.logical.OpAx
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.math.drm.logical.OpAt
import org.apache.mahout.math.drm.logical.OpAtx
import org.apache.mahout.math.drm.logical.OpAtx
import org.apache.mahout.math.drm.logical.OpAtB
import org.apache.mahout.math.drm.logical.OpABt
import org.apache.mahout.math.drm.logical.OpAtB
import org.apache.mahout.math.drm.logical.OpAtA
import org.apache.mahout.math.drm.logical.OpAewScalar
import org.apache.mahout.math.drm.logical.OpAewB
import org.apache.mahout.math.drm.logical.OpCbind
import org.apache.mahout.math.drm.logical.OpRbind
import org.apache.mahout.math.drm.logical.OpMapBlock
import org.apache.mahout.math.drm.logical.OpRowRange
import org.apache.mahout.math.drm.logical.OpTimesRightMatrix

object FlinkEngine extends DistributedEngine {

  /** Second optimizer pass. Translate previously rewritten logical pipeline into physical engine plan. */
  override def toPhysical[K: ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] = {
    // Flink-specific Physical Plan translation.
    val drm = flinkTranslate(plan)

    val newcp = new CheckpointedFlinkDrm(
      ds = drm.deblockify.ds,
      _nrow = plan.nrow,
      _ncol = plan.ncol
    )

    newcp.cache()
  }

  private def flinkTranslate[K: ClassTag](oper: DrmLike[K]): FlinkDrm[K] = oper match {
    case op @ OpAx(a, x) => FlinkOpAx.blockifiedBroadcastAx(op, flinkTranslate(a)(op.classTagA))
    case op @ OpAt(a) => FlinkOpAt.sparseTrick(op, flinkTranslate(a)(op.classTagA))
    case op @ OpAtx(a, x) => {
      // express Atx as (A.t) %*% x
      // TODO: create specific implementation of Atx
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
      // TODO: create specific implementation of ABt
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
      // TODO: create specific implementation of AtA
      val aInt = a.asInstanceOf[DrmLike[Int]] // TODO: casts!
      val opAtB = OpAtB(aInt, aInt)
      val aTranslated = flinkTranslate(aInt)
      FlinkOpAtB.notZippable(opAtB, aTranslated, aTranslated)
    }
    case op @ OpTimesRightMatrix(a, b) => 
      FlinkOpTimesRightMatrix.drmTimesInCore(op, flinkTranslate(a)(op.classTagA), b)
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
  

  def translate[K: ClassTag](oper: DrmLike[K]): DataSet[K] = ???

  /** Engine-specific colSums implementation based on a checkpoint. */
  override def colSums[K: ClassTag](drm: CheckpointedDrm[K]): Vector = ???

  /** Engine-specific numNonZeroElementsPerColumn implementation based on a checkpoint. */
  override def numNonZeroElementsPerColumn[K: ClassTag](drm: CheckpointedDrm[K]): Vector = ???

  /** Engine-specific colMeans implementation based on a checkpoint. */
  override def colMeans[K: ClassTag](drm: CheckpointedDrm[K]): Vector = ???

  override def norm[K: ClassTag](drm: CheckpointedDrm[K]): Double = ???

  /** Broadcast support */
  override def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] = ???

  /** Broadcast support */
  override def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] = ???

  /**
   * Load DRM from hdfs (as in Mahout DRM format).
   * <P/>
   * @param path The DFS path to load from
   * @param parMin Minimum parallelism after load (equivalent to #par(min=...)).
   */
  override def drmDfsRead(path: String, parMin: Int = 0)
                         (implicit sc: DistributedContext): CheckpointedDrm[_] = ???

  /** Parallelize in-core matrix as spark distributed matrix, using row ordinal indices as data set keys. */
  override def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)
                                           (implicit sc: DistributedContext): CheckpointedDrm[Int] = {
    val parallelDrm = parallelize(m, numPartitions)
    new CheckpointedFlinkDrm(ds=parallelDrm, _nrow=m.numRows(), _ncol=m.numCols())
  }

  private[flinkbindings] def parallelize(m: Matrix, parallelismDegree: Int)
      (implicit sc: DistributedContext): DrmDataSet[Int] = {
    val rows = (0 until m.nrow).map(i => (i, m(i, ::)))
    val rowsJava: Collection[DrmTuple[Int]]  = rows.asJava

    val dataSetType = TypeExtractor.getForObject(rows.head)
    sc.env.fromCollection(rowsJava, dataSetType).setParallelism(parallelismDegree)
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
   * Load IndexedDataset from text delimited format.
   * @param src comma delimited URIs to read from
   * @param schema defines format of file(s)
   */
  override def indexedDatasetDFSRead(src: String,
                   schema: Schema = DefaultIndexedDatasetReadSchema, 
                   existingRowIDs: BiMap[String, Int] = HashBiMap.create())
            (implicit sc: DistributedContext): IndexedDataset = ???

  /**
   * Load IndexedDataset from text delimited format, one element per line
   * @param src comma delimited URIs to read from
   * @param schema defines format of file(s)
   */
  override def indexedDatasetDFSReadElements(src: String,
                    schema: Schema = DefaultIndexedDatasetElementReadSchema,
                    existingRowIDs: BiMap[String, Int] = HashBiMap.create())
             (implicit sc: DistributedContext): IndexedDataset = ???
}