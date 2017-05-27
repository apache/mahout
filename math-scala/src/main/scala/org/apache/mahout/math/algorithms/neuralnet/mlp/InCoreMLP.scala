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

package org.apache.mahout.math.algorithms.neuralnet.mlp

import org.apache.mahout.math.{Matrices, Matrix, Vector}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.function.{DoubleFunction, VectorFunction}

import collection._
import JavaConversions._
import org.apache.mahout.math.function.Functions._

/**
  * Creates an InCoreMLP (Multilayer Perceptron), a simple sort of neural network.
  *
  */
class InCoreMLP extends Serializable {

  var A: Array[Vector] = _
  var architecture: Vector = _
  var convergenceThreshold: Double = 0.01
  var delta: Array[Vector] = _
  var gradient: Array[Matrix] = _
  var hasConverged: Boolean = false
  var initialLearningRate: Double = 0.01
  var iteration: Int = 0
  var L: Int = _
  var targetIsDelta: Boolean = false
  var U: Array[_ <: Matrix] = _
  var Z: Array[Vector] = _

  // These are must be set to full Vectors/Matrices
  var inputStart: Int = 0
  var inputOffset: Int = _
  var targetStart: Int = _
  var targetOffset: Int = _

  var activationFn : DoubleFunction = SIGMOID
  var activationFnDerivative : DoubleFunction = SIGMOIDGRADIENT
  /**
    *
    * @param arch         A [[org.apache.mahout.math.Vector]] specifying the architecture of the neural network. The first number is the expected
    *                     number of input, the last number is the expected number of outputs or targets. The numbers in
    *                     between specify how many nodes on the hidden layers.
    * @return             none, initializes weights (U) for the neural network.
    */
  def createWeightsArray(arch: Vector): Unit = {
    architecture = arch
    U = (0 until architecture.size()-1)
      .map(i => Matrices.gaussianView(architecture.get(i+1).toInt, architecture.get(i).toInt, 1).cloned ).toArray

  }

  /**
    * Calculate the forward-pass of the network
    * @param x the input Vector
    * @return
    */
  def feedForward(x: Vector): (Array[Vector], Array[Vector]) = {
    A = Array(x.cloned) ++ new Array[Vector](U.size)
    Z = Array(x.cloned) ++ new Array[Vector](U.size)
    for (l <- 1 until A.size){
      A(l) = (U(l - 1) %*% A(l-1)) // + TODO: biases ...
      Z(l) = A(l).cloned.assign(activationFn)
    }
    (A, Z)
  }

  /**
    * Back propegate the error through the network creating the gradient
    * @param target Either the target (if this is the top level of the netowrk) or the delta (or part of the delta) if
    *               there is a "higher network" this is back propegating through (e.g. convelution networks)
    * @return none, updates gradient for this portion of the network.
    */
  def backProp(target: Vector): Unit = {
    L = A.size - 1 // move this outside
    delta = new Array[Vector](A.size)
    gradient = new Array[Matrix](U.size)

    val error = A(L) - target
    if (error.sum < convergenceThreshold){
      hasConverged = true
    }

    // This allows for us to "bolt" neural nets together and back prop across multiple
    delta(L) = targetIsDelta match {
      case false => error * Z(L).assign(activationFnDerivative) // todo replace (A(L) - target) with LossFnDerivative
      case true => target
    }

    gradient(L-1) = delta(L).cross(A(L-1))
    for (l <- L -1 to 1 by -1){
      delta(l) = (U(l).t %*% delta(l + 1)) * Z(l).assign(activationFnDerivative)
      gradient(l-1) = delta(l).cross(A(l-1))
    }

  }

  def updateU(): Unit = {
    val learningRate = initialLearningRate // todo create learning rate function, fn of iteration
    U = (0 until U.size).map(l => U(l) - (learningRate * gradient(l))).toArray
    iteration += 1
  }

  /**
    * A convenience method for doing a full forward-backward pass and updating the weight vector
    * (can't use this when building complex networks- for simplenetworks only.)
    * @param x [[org.apache.mahout.math.Vector]] the input
    * @param target [[org.apache.mahout.math.Vector]] the target
    */
  def forwardBackward(x: Vector, target: Vector): Unit = {
    feedForward(x)
    backProp(target)
    updateU()
  }

  def forwardBackwardVector(v: Vector): Unit = {
    val x = v.viewPart(inputStart, inputOffset)
    val y = v.viewPart(targetStart, targetOffset).cloned
    forwardBackward(x, y)
  }

  def forwardBackwardMatrix(m: Matrix): Unit = {
    import collection._
    import JavaConversions._

    for (v <- m){
      forwardBackwardVector(v)
    }
  }

  def parameterVector(): Vector = {
    import org.apache.mahout.math.algorithms.neuralnet.Converters
    Converters.flattenMatrixArrayToVector(U)
  }


  def setParametersFromVector(v: Vector): Unit ={
    import org.apache.mahout.math.algorithms.neuralnet.Converters
    val sizeArray = new Array[(Int, Int)](architecture.length - 1)
    for (i <- 0 until (architecture.length - 1)){
      sizeArray(i) = (architecture(i + 1).toInt, architecture(i).toInt)
    }
    U = Converters.recomposeMatrixArrayFromVec(v, sizeArray)
  }
}


