/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.math.algorithms

import org.apache.mahout.math.algorithms.neuralnet.mlp.InCoreMLP
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.test.MahoutSuite
import org.scalatest.FunSuite

class NeuralNetSuite extends FunSuite with MahoutSuite{

  val epsilon = 1E-6

  test("feed forward"){
    val testMLP = new InCoreMLP()
    testMLP.U = Array( dense((0, 0), (1, 0), (0, 0)), dense(0, 1, 0).t )
    testMLP.feedForward( (1, 0) )
    (testMLP.A(1).get(0) - 0.7310585786300049) should be < epsilon

    testMLP.useBiases = true
    testMLP.biases = Array( dense((0.1, 0.1)), dense((0.1, 0.1, 0.1)), dense((0.1)) )
    testMLP.feedForward( (1, 0) )
    (testMLP.A(1).get(0) - 0.7685247834990178) should be < epsilon
  }

  test("back prop") {
    val testMLP = new InCoreMLP()
    testMLP.createWeightsArray( dvec(2, 3, 1) )
    testMLP.U = Array( dense((1, 0), (0, 0), (1, 0)), dense( 1, 0, 1).t )
    testMLP.feedForward( (1, 0) )
    testMLP.backProp(dvec(1))
    testMLP.gradient(0).get(0, 0) should equal (testMLP.gradient(0).get(2, 0))
  }


  // todo add tests for learning rate, etc once they are added.

}
