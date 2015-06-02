package org.apache.mahout

import scala.reflect.ClassTag
import org.slf4j.LoggerFactory
import org.apache.flink.api.java.DataSet
import org.apache.flink.api.java.ExecutionEnvironment
import org.apache.flink.api.common.functions.MapFunction
import org.apache.mahout.math.Vector
import org.apache.mahout.math.DenseVector
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.MatrixWritable
import org.apache.mahout.math.VectorWritable
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.flinkbindings.FlinkDistributedContext
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.flinkbindings.drm.CheckpointedFlinkDrm
import org.apache.flink.api.common.functions.FilterFunction

package object flinkbindings {

  private[flinkbindings] val log = LoggerFactory.getLogger("apache.org.mahout.flinkbingings")

  /** Row-wise organized DRM dataset type */
  type DrmDataSet[K] = DataSet[DrmTuple[K]]

  /**
   * Blockifed DRM dataset (keys of original DRM are grouped into array corresponding to rows of Matrix
   * object value
   */
  type BlockifiedDrmDataSet[K] = DataSet[BlockifiedDrmTuple[K]]

  
  implicit def wrapMahoutContext(context: DistributedContext): FlinkDistributedContext = {
    assert(context.isInstanceOf[FlinkDistributedContext], "it must be FlinkDistributedContext")
    context.asInstanceOf[FlinkDistributedContext]
  }

  implicit def wrapContext(env: ExecutionEnvironment): FlinkDistributedContext =
    new FlinkDistributedContext(env)
  implicit def unwrapContext(ctx: FlinkDistributedContext): ExecutionEnvironment = ctx.env

  private[flinkbindings] implicit def castCheckpointedDrm[K: ClassTag](drm: CheckpointedDrm[K]): CheckpointedFlinkDrm[K] = {
    assert(drm.isInstanceOf[CheckpointedFlinkDrm[K]], "it must be a Flink-backed matrix")
    drm.asInstanceOf[CheckpointedFlinkDrm[K]]
  }

  implicit def checkpointeDrmToFlinkDrm[K: ClassTag](cp: CheckpointedDrm[K]): FlinkDrm[K] = {
    val flinkDrm = castCheckpointedDrm(cp)
    new RowsFlinkDrm[K](flinkDrm.ds, flinkDrm.ncol)
  }

  private[flinkbindings] implicit def wrapAsWritable(m: Matrix): MatrixWritable = new MatrixWritable(m)
  private[flinkbindings] implicit def wrapAsWritable(v: Vector): VectorWritable = new VectorWritable(v)
  private[flinkbindings] implicit def unwrapFromWritable(w: MatrixWritable): Matrix = w.get()
  private[flinkbindings] implicit def unwrapFromWritable(w: VectorWritable): Vector = w.get()

  def readCsv(file: String, delim: String = ",", comment: String = "#")
             (implicit dc: DistributedContext): CheckpointedDrm[Long] = {
    val vectors = dc.env.readTextFile(file)
        .filter(new FilterFunction[String] {
          def filter(in: String): Boolean = {
            !in.startsWith(comment)
          }
        })
        .map(new MapFunction[String, Vector] {
          def map(in: String): Vector = {
            val array = in.split(delim).map(_.toDouble)
            new DenseVector(array)
          }
        })
    datasetToDrm(vectors)
  }

  def datasetToDrm(ds: DataSet[Vector]): CheckpointedDrm[Long] = {
    val zipped = new DataSetOps(ds).zipWithIndex
    datasetWrap(zipped)
  }

  def datasetWrap[K: ClassTag](dataset: DataSet[(K, Vector)]): CheckpointedDrm[K] = {
    new CheckpointedFlinkDrm[K](dataset)
  }


}