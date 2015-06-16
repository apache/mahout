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

object FlinkEngine extends DistributedEngine {

  /** Second optimizer pass. Translate previously rewritten logical pipeline into physical engine plan. */
  override def toPhysical[K: ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] = {
    // Flink-specific Physical Plan translation.
    val drm = flinkTranslate(plan)

    val newcp = new CheckpointedFlinkDrm(
      ds = drm.deblockify.ds, // TODO: make it lazy!
      _nrow = plan.nrow,
      _ncol = plan.ncol
//      _cacheStorageLevel = cacheHint2Spark(ch),
//      partitioningTag = plan.partitioningTag
    )

    newcp.cache()
  }

  private def flinkTranslate[K: ClassTag](oper: DrmLike[K]): FlinkDrm[K] = oper match {
    case op @ OpAx(a, x) => FlinkOpAx.blockifiedBroadcastAx(op, flinkTranslate(a)(op.classTagA))
    case op @ OpAt(a) => FlinkOpAt.sparseTrick(op, flinkTranslate(a)(op.classTagA))
    case op @ OpAtx(a, x) => {
      val opAt = OpAt(a)
      val at = FlinkOpAt.sparseTrick(opAt, flinkTranslate(a)(op.classTagA))
      val atCast = new CheckpointedFlinkDrm(at.deblockify.ds, _nrow=opAt.nrow, _ncol=opAt.ncol)
      val opAx = OpAx(atCast, x)
      FlinkOpAx.blockifiedBroadcastAx(opAx, flinkTranslate(atCast)(op.classTagA))
    }
    case cp: CheckpointedFlinkDrm[K] => new RowsFlinkDrm(cp.ds, cp.ncol)
    case _ => ???
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