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

package org.apache.mahout.sparkbindings.algorithms.regression

import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.algorithms._
import org.apache.mahout.math.algorithms.regression._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{Vector => SparkMLVector}

/**
 * A wrapper to expose the OrdinaryLeastSquares algorithm from mahout
 * in Spark's pipeline interface.
 */
class OrdinaryLeastSquaresEstimator(override val uid: String)
    extends SupervisedSparkEstimator[
      OrdinaryLeastSquaresModel[Long],
      OrdinaryLeastSquares[Long],
      OrdinaryLeastSquaresPipelineModel]
// TODO(provide these)
//    with HasCalcCommonStatistics with HasCalcStandardErrors with HasAddIntercept {
{
  def this() = this(Identifiable.randomUID("OrdinaryLeastSquaresEstimator"))

  def constructSupervisedMahoutFitter() = {
    new OrdinaryLeastSquares[Long]()
  }

  def constructSparkModel(model: OrdinaryLeastSquaresModel[Long]) = {
    new OrdinaryLeastSquaresPipelineModel(model)
  }

  override def copy(extra: ParamMap) = {
    defaultCopy(extra)
  }
}

class OrdinaryLeastSquaresPipelineModel private (
  override val uid: String,
  private val model: OrdinaryLeastSquaresModel[Long])
    extends SparkPredictorModel[
      OrdinaryLeastSquaresPipelineModel,
      OrdinaryLeastSquaresModel[Long]] {

  private[mahout] def this(model: OrdinaryLeastSquaresModel[Long]) = {
    this(Identifiable.randomUID("OLSPM"), model)
  }

  override def copy(extra: ParamMap) = {
    val copied = new OrdinaryLeastSquaresPipelineModel(uid, model)
    copyValues(copied, extra)
  }

  /**
   * Override transformImpl so we can convert the input into a DRM like
   */
  override protected def transformImpl(dataset: Dataset[_]) = {
    val session = dataset.sparkSession
    import session.implicits._
    val ds = dataset.select(
      dataset("*"),
      monotonically_increasing_id().as("_temporary_id"),
      dataset($(featuresCol)))
    val drmInput: DrmLike[Long] = drmWrapDataFrameML(ds)
    val predictedDrm = model.predict(drmInput)
    val predictedRDD: RDD[(Long, Double)] = predictedDrm match {
      case x: DrmRdd[Long] =>
          x.map{case (id, mahoutRow) =>
            (id, mahoutRow.get(0))
          }
      // In theory we can't have a blockified drm since long keys.
      case _ =>
        throw new Exception("Unsupported DrmLike type returned. Expected DrmRdd or BlockifiedDrmRdd got ${predict}")
    }
    val predictedDS = predictedRDD.toDF("_temporary_id", $(predictionCol))
    val columnNames = predictedDS.columns.filter(_ != "_temporary_id")
    predictedDS.join(ds, "_temporary_id").select(columnNames.head, columnNames.tail:_*)
  }

  /**
   * This is a little bad, we don't actually implement predict but it is protected
   * and we override the only case where it is called.
   */
  override protected def predict(features: SparkMLVector): Double = {
    throw new Exception("Predict is not supported, use transform.")
  }
}
