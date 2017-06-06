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
package org.apache.mahout

import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala.utils._
import org.apache.flink.api.scala.{DataSet, ExecutionEnvironment, _}
import org.apache.mahout.flinkbindings.drm.{CheckpointedFlinkDrm, CheckpointedFlinkDrmOps, FlinkDrm, RowsFlinkDrm}
import org.apache.mahout.math.drm.{BlockifiedDrmTuple, CheckpointedDrm, DistributedContext, DrmTuple, _}
import org.apache.mahout.math.{DenseVector, Matrix, MatrixWritable, Vector, VectorWritable}
import org.slf4j.LoggerFactory

import scala.Array._
import scala.reflect.ClassTag

package object flinkbindings {

  private[flinkbindings] val log = LoggerFactory.getLogger("org.apache.mahout.flinkbindings")

  /** Row-wise organized DRM dataset type */
  type DrmDataSet[K] = DataSet[DrmTuple[K]]

  /**
   * Blockified DRM dataset (keys of original DRM are grouped into array corresponding to rows of Matrix
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

  private[flinkbindings] implicit def castCheckpointedDrm[K: ClassTag](drm: CheckpointedDrm[K])
    : CheckpointedFlinkDrm[K] = {

    assert(drm.isInstanceOf[CheckpointedFlinkDrm[K]], "it must be a Flink-backed matrix")
    drm.asInstanceOf[CheckpointedFlinkDrm[K]]
  }

  implicit def checkpointedDrmToFlinkDrm[K: TypeInformation: ClassTag](cp: CheckpointedDrm[K]): FlinkDrm[K] = {
    val flinkDrm = castCheckpointedDrm(cp)
    new RowsFlinkDrm[K](flinkDrm.ds, flinkDrm.ncol)
  }

  /** Adding Flink-specific ops */
  implicit def cpDrm2cpDrmFlinkOps[K: ClassTag](drm: CheckpointedDrm[K]): CheckpointedFlinkDrmOps[K] =
    new CheckpointedFlinkDrmOps[K](drm)

  implicit def drm2cpDrmFlinkOps[K: ClassTag](drm: DrmLike[K]): CheckpointedFlinkDrmOps[K] = drm: CheckpointedDrm[K]


  private[flinkbindings] implicit def wrapAsWritable(m: Matrix): MatrixWritable = new MatrixWritable(m)
  private[flinkbindings] implicit def wrapAsWritable(v: Vector): VectorWritable = new VectorWritable(v)
  private[flinkbindings] implicit def unwrapFromWritable(w: MatrixWritable): Matrix = w.get()
  private[flinkbindings] implicit def unwrapFromWritable(w: VectorWritable): Vector = w.get()


  def readCsv(file: String, delim: String = ",", comment: String = "#")
             (implicit dc: DistributedContext): CheckpointedDrm[Long] = {
    val vectors = dc.env.readTextFile(file)
      .filter((in: String) => {
        !in.startsWith(comment)
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
    val zipped = ds.zipWithIndex
    datasetWrap(zipped)
  }

  def datasetWrap[K: ClassTag](dataset: DataSet[(K, Vector)]): CheckpointedDrm[K] = {
    implicit val typeInformation = FlinkEngine.generateTypeInformation[K]
    new CheckpointedFlinkDrm[K](dataset)
  }

  private[flinkbindings] def extractRealClassTag[K: ClassTag](drm: DrmLike[K]): ClassTag[_] = drm.keyClassTag

  private[flinkbindings] def getMahoutHome() = {
    var mhome = System.getenv("MAHOUT_HOME")
    if (mhome == null) mhome = System.getProperty("mahout.home")
    require(mhome != null, "MAHOUT_HOME is required to spawn mahout-based flink jobs")
    mhome
  }
}