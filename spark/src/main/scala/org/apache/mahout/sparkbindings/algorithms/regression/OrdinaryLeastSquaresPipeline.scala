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

package org.apache.mahout.spark.sparkbindings.algorithms.regression

import org.apache.mahout.spark.sparkbindings.algorithms._
import org.apache.mahout.math.algorithms.regression._

/**
 * A wrapper to expose the OrdinaryLeastSquares algorithm from mahout
 * in Spark's pipeline interface.
 */
class OrdinaryLeastSquaresEstimator
    extends SupervisedSparkEstimator[
      OrdinaryLeastSquares[Double], OrdinaryLeastSquaresModel[Double],
      OrdinaryLeastSquaresPipelineModel]
// TODO(provide these)
//    with HasCalcCommonStatistics with HasCalcStandardErrors with HasAddIntercept {
{
}

class OrdinaryLeastSquaresPipelineModel(val model: OrdinaryLeastSquaresModel[Double]) extends
    SparkModel[OrdinaryLeastSquaresModel[Double]] {
}
