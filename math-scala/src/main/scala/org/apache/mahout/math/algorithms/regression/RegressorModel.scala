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

package org.apache.mahout.math.algorithms.regression

import org.apache.mahout.math.algorithms.{SupervisedFitter, SupervisedModel}
import org.apache.mahout.math.drm.DrmLike

trait RegressorModel[K] extends SupervisedModel[K] {

  def predict(drmPredictors: DrmLike[K]): DrmLike[K]

  var addIntercept: Boolean = _
  // Common Applicable Tests- here only for convenience.
  var mse: Double = _
  var r2: Double = _
  var fpval: Double = _
  // default rss to a negative number to ensure rss gets set.
  var rss:Double = -9999.0
  var fScore: Double = _
  var degreesOfFreedom: Int = _
  var trainingExamples :Int = _

  /**
    * Syntatictic sugar for fetching test results.  Will Return test result if it exists, otherwise None
    * @param testSymbol - symbol of the test result to fetch, e.g. `'mse`
    * @tparam T - The Type
    * @return
    */
  def getTestResult[T](testSymbol: Symbol): Option[T] = {
    Some(testResults.get(testSymbol).asInstanceOf[T])
  }
}

trait RegressorFitter[K] extends SupervisedFitter[K, RegressorModel[K]] {

  var addIntercept: Boolean = _

  def fitPredict(drmX: DrmLike[K],
                 drmTarget: DrmLike[K],
                 hyperparameters: (Symbol, Any)* ): DrmLike[K] = {

    model = this.fit(drmX, drmTarget, hyperparameters: _* )
    model.predict(drmX)
  }

  // used to store the model if `fitTransform` method called
  var model: RegressorModel[K] = _

}
