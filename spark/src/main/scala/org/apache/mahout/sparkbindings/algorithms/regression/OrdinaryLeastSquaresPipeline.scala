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

import org.apache.mahout.sparkbindings.algorithms._
import org.apache.mahout.math.algorithms.regression._

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.ParamMap

/**
 * A wrapper to expose the OrdinaryLeastSquares algorithm from mahout
 * in Spark's pipeline interface.
 */
/*
class OrdinaryLeastSquaresEstimator(override val uid: String)
    extends SupervisedSparkEstimator[
      OrdinaryLeastSquaresModel[Double],
      OrdinaryLeastSquares[Double],
      OrdinaryLeastSquaresPipelineModel]
// TODO(provide these)
//    with HasCalcCommonStatistics with HasCalcStandardErrors with HasAddIntercept {
{
  def this() = this(Identifiable.randomUID("OrdinaryLeastSquaresEstimator"))

  def constructSupervisedMahoutFitter() = {
    new OrdinaryLeastSquares[Double]()
  }

  def constructSparkModel(model: OrdinaryLeastSquaresModel[Double]) = {
    new OrdinaryLeastSquaresPipelineModel(model)
  }

  override def copy(extra: ParamMap) = {
    defaultCopy(extra)
  }
}

class OrdinaryLeastSquaresPipelineModel(
  override val uid: String,
  val model: OrdinaryLeastSquaresModel[Double]) extends
    SparkModel[OrdinaryLeastSquaresModel[Double]] {
  private[mahout] def this(model: OrdinaryLeastSquaresModel[Double]) = {
    this(Identifiable.randomUID("OLSPM"), model)
  }

  override def copy(extra: ParamMap) = {
    val copied = new OrdinaryLeastSquaresPipelineModel(uid, model)
    copyValues(copied, extra)
  }
}
*/
