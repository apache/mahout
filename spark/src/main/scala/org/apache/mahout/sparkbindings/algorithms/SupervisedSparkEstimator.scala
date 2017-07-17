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

package org.apache.mahout.spark.sparkbindings.algorithms

import org.apache.mahout.math.algorithms._

import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.sql.{Dataset, SparkSession}

class SupervisedSparkEstimator[
  M <: SupervisedModel[Long],
  F <: SupervisedFitter[Long, M],
  S <: SparkModel[M]] extends SparkEstimator[S]
    with HasFeaturesCol with HasLabelCol with HasOutputCol {
  override def fit(ds: Dataset[_], hyperparameters: (Symbol, Any)*): S = {
    // We would use TypedColumns here except Spark's VectorUDT is private because of "reasons".
    val sparkInput = ds.select(
      monotonicallyIncreasingId(), ds($(labelCol)), ds($(featuresCol))).rdd.persist()
    val labels: RDD[(Long, Double)] = sparkInput.map{row => (row.getLong(0), row.getDouble(1))}
    val features: RDD[(Long, SparkVector)] = sparkInput.map{row => (row.getLong(0), row.get(2).asInstanceOf[SparkVector])}
    val fitter = constructSupervisedMahoutFitter()
    // For some reason we index these?
    val mahoutLabels: DrmRdd[Long] = drmWrap[](labels)
    val mahoutFeatures: DrmRdd[Long] = drmWrapSparkMLVector(features)
    val model = constructSparkModel(
      fitter.fit(mahoutLabels, mahoutFeatures, hyperparameters))
    sparkInput.unpersist()
    model
  }

  def constructSupervisedMahoutFitter(): F

  def constructSparkModel(mahoutModel: M): S
}
