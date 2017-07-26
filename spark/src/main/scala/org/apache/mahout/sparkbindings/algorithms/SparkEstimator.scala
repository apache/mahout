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

import org.apache.spark.ml.{Estimator, Model => SModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset

/**
 * This is a general purpose base trait for helping implement different bridges in Mahout.
 * This should not be assumed to be the root trait since we often need to inherit from a specific
 * subtrait in Spark that may conflict, so some estimators will not have this as a parent.
 */
trait SparkEstimator[M <: MModel, S <: SModel[S]] extends Estimator[S] with HasOutputCol {
  /**
   * Fit your model. This helper function extracts the parameter map
   * and converts it to the Mahout hyperparameters for interop.
   */
  override def fit(ds: Dataset[_]): S = {
    val hyperparameters = extractParamMap(ParamMap.empty).toSeq.map{
      paramPair => (Symbol.apply(paramPair.param.name), paramPair.value)
    }
    fit(ds, hyperparameters:_*)
  }

  /**
   * Implement this method to receive the hyper parameters and perform the fitting.
   */
  def fit(ds: Dataset[_], hyperparameters: (Symbol, Any)*): S
}
