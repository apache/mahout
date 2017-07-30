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

package org.apache.mahout.sparkbindings.algorithms

import org.apache.mahout.math.{Vector => MahoutVector}
import org.apache.mahout.math.algorithms._
import org.apache.mahout.sparkbindings._
import org.apache.mahout.spark.sparkbindings._
import org.apache.mahout.math.algorithms.regression._

import org.apache.spark.rdd._
import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Predictor, PredictionModel}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

trait SupervisedSparkEstimator[
  M <: SupervisedModel[Long],
  F <: SupervisedFitter[Long, _],
  S <: SparkPredictorModel[S, M]]
    extends Predictor[SparkVector, SupervisedSparkEstimator[M, F, S], S] {

  override def train(ds: Dataset[_]): S = {
   val hyperparameters = extractParamMap(ParamMap.empty).toSeq.map{
      paramPair => (Symbol.apply(paramPair.param.name), paramPair.value)
    }

    // We would use TypedColumns here except Spark's VectorUDT is private because of "reasons".
    val sparkInput = ds.select(
      monotonically_increasing_id(), ds($(labelCol)), ds($(featuresCol))).rdd.persist()

    // Compute the number of rows and number of columns of the features
    val nrow = sparkInput.count()
    val ncol = sparkInput.take(1).headOption.map{
      vec => vec.get(2).asInstanceOf[SparkVector].size}.getOrElse(0)

    // Extract the labels and features as separate RDDs
    val labels: RDD[(Long, Double)] = sparkInput.map{row => (row.getLong(0), row.getDouble(1))}
    val features: RDD[(Long, SparkVector)] = sparkInput.map{row => (row.getLong(0), row.get(2).asInstanceOf[SparkVector])}

    // Convert the labels and features into Mahout's internal format
    val labelsDrmRdd = labels.mapValues(v =>
      new org.apache.mahout.math.DenseVector(Array(v)).asInstanceOf[MahoutVector])
    val mahoutLabels = drmWrap[Long](
      labelsDrmRdd, nrow = nrow, ncol = 1, canHaveMissingRows = true)
    val mahoutFeatures = drmWrapSparkMLVector[Long](
      rdd = features, nrow = nrow, ncol = ncol, canHaveMissingRows = true)

    // Fit the mahout model and wrap it.
    val fitter = constructSupervisedMahoutFitter()
    val mahoutModel = fitter.fit(mahoutLabels, mahoutFeatures, hyperparameters:_*)
    val model = constructSparkModel(mahoutModel.asInstanceOf[M])
    sparkInput.unpersist()
    copyValues(model)
  }

  def constructSupervisedMahoutFitter(): F

  def constructSparkModel(mahoutModel: M): S
}
