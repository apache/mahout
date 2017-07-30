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

import org.apache.mahout.math.algorithms.{Model => MModel}

import org.apache.spark.ml.{Model => SModel, PredictionModel}
import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.sql.{Dataset, DataFrame}

/**
 * Common Spark Model created around a Mahout model.
 */
trait SparkModel[T <: SparkModel[T, M], M <: MModel]
    extends SModel[T] with HasOutputCol {
}

/**
 * Spark Predictor Model.
 */
trait SparkPredictorModel[T <: SparkPredictorModel[T, M], M <: MModel]
    extends PredictionModel[SparkVector, T] with SparkModel[T, M] {
}
