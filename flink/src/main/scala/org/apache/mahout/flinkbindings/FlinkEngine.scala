package org.apache.mahout.flinkbindings

import scala.reflect.ClassTag

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

import com.google.common.collect.BiMap
import com.google.common.collect.HashBiMap

object FlinkEngine extends DistributedEngine {

  /** Second optimizer pass. Translate previously rewritten logical pipeline into physical engine plan. */
  override def toPhysical[K: ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] = {
    null
  }

  /** Engine-specific colSums implementation based on a checkpoint. */
  override def colSums[K: ClassTag](drm: CheckpointedDrm[K]): Vector = {
    null
  }

  /** Engine-specific numNonZeroElementsPerColumn implementation based on a checkpoint. */
  override def numNonZeroElementsPerColumn[K: ClassTag](drm: CheckpointedDrm[K]): Vector = {
    null
  }

  /** Engine-specific colMeans implementation based on a checkpoint. */
  override def colMeans[K: ClassTag](drm: CheckpointedDrm[K]): Vector = {
    null
  }

  override def norm[K: ClassTag](drm: CheckpointedDrm[K]): Double = {
    0.0d
  }

  /** Broadcast support */
  override def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] = {
    null
  }

  /** Broadcast support */
  override def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] = {
    null
  }

  /**
   * Load DRM from hdfs (as in Mahout DRM format).
   * <P/>
   * @param path The DFS path to load from
   * @param parMin Minimum parallelism after load (equivalent to #par(min=...)).
   */
  override def drmDfsRead(path: String, parMin: Int = 0)
                         (implicit sc: DistributedContext): CheckpointedDrm[_] = {
    null
  }

  /** Parallelize in-core matrix as spark distributed matrix, using row ordinal indices as data set keys. */
  override def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)
                                           (implicit sc: DistributedContext): CheckpointedDrm[Int] = {
    null
  }

  /** Parallelize in-core matrix as spark distributed matrix, using row labels as a data set keys. */
  override def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int = 1)
                                          (implicit sc: DistributedContext): CheckpointedDrm[String] = {
    null
  }

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  override def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int = 10)
                                  (implicit sc: DistributedContext): CheckpointedDrm[Int] = {
    null
  }

  /** Creates empty DRM with non-trivial height */
  override def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int = 10)
                                      (implicit sc: DistributedContext): CheckpointedDrm[Long] = {
    null
  }

  /**
   * Load IndexedDataset from text delimited format.
   * @param src comma delimited URIs to read from
   * @param schema defines format of file(s)
   */
  override def indexedDatasetDFSRead(src: String,
                   schema: Schema = DefaultIndexedDatasetReadSchema, 
                   existingRowIDs: BiMap[String, Int] = HashBiMap.create())
            (implicit sc: DistributedContext): IndexedDataset = {
    null
  }

  /**
   * Load IndexedDataset from text delimited format, one element per line
   * @param src comma delimited URIs to read from
   * @param schema defines format of file(s)
   */
  override def indexedDatasetDFSReadElements(src: String,
                    schema: Schema = DefaultIndexedDatasetElementReadSchema,
                    existingRowIDs: BiMap[String, Int] = HashBiMap.create())
             (implicit sc: DistributedContext): IndexedDataset = {
    null
  }
}