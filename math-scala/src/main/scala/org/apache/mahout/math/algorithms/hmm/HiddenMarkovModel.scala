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

package org.apache.mahout.math.algorithms.hmm

import scala.util.Random
import org.apache.mahout.math.algorithms.{UnsupervisedFitter, UnsupervisedModel}
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.function.VectorFunction
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.common.RandomUtils

object HMMFunctions {
  def computeForwardVariables(numberOfHiddenStates: Int,
    initialProbabilities: Vector,
    transitionMatrix: Matrix,
    emissionMatrix: Matrix,
    observationSequence:Vector,
    scale: Boolean
  ): (DenseMatrix, Option[Array[Double]]) = {
    var forwardVariables = new DenseMatrix(observationSequence.size, numberOfHiddenStates)
    var scalingFactors = None: Option[Array[Double]]

    if (scale) {
      scalingFactors = Some(new Array(observationSequence.size))
      var forwardVariablesTemp = new DenseMatrix(observationSequence.size, numberOfHiddenStates)
      // Initialization
      for (index <- 0 until numberOfHiddenStates) {
        forwardVariablesTemp(0, index) = initialProbabilities(index) * emissionMatrix(index, observationSequence(0).toInt)
      }

      var sum:Double = 0.0
      for (index <- 0 until numberOfHiddenStates) {
        sum += forwardVariablesTemp(0, index)
      }

      scalingFactors.get(0) = 1.0/sum

      for (index <- 0 until numberOfHiddenStates) {
        forwardVariables(0, index) = forwardVariablesTemp(0, index) * scalingFactors.get(0)
      }

      // Induction
      for (indexT <- 1 until observationSequence.size) {
        for (indexN <- 0 until numberOfHiddenStates) {
	  var sumA:Double = 0.0
	  for (indexM <- 0 until numberOfHiddenStates) {
            sumA += forwardVariables(indexT - 1, indexM) * transitionMatrix(indexM, indexN) * emissionMatrix(indexN, observationSequence(indexT).toInt)
          }

          forwardVariablesTemp(indexT, indexN) = sumA
        }

        var sumT:Double = 0.0
        for (indexN <- 0 until numberOfHiddenStates) {
          sumT += forwardVariablesTemp(indexT, indexN)
        }

        scalingFactors.get(indexT) = 1.0/sumT

        for (indexN <- 0 until numberOfHiddenStates) {
          forwardVariables(indexT, indexN)= scalingFactors.get(indexT) * forwardVariablesTemp(indexT, indexN)
        }
      }
    } else {
      // Initialization
      for (index <- 0 until numberOfHiddenStates) {
        forwardVariables(0, index) = initialProbabilities(index) * emissionMatrix(index, observationSequence(0).toInt)
      }

      // Induction
      for (indexT <- 1 until observationSequence.size) {
        for (indexN <- 0 until numberOfHiddenStates) {
	  var sum:Double = 0.0
	  for (indexM <- 0 until numberOfHiddenStates) {
	    sum += forwardVariables(indexT - 1, indexM) * transitionMatrix(indexM, indexN);
	  }

	  forwardVariables(indexT, indexN) = sum * emissionMatrix(indexN, observationSequence(indexT).toInt)
	}
      }
    }

    (forwardVariables, scalingFactors)
  }

  def computeBackwardVariables(numberOfHiddenStates: Int,
    initialProbabilities: Vector,
    transitionMatrix: Matrix,
    emissionMatrix: Matrix,
    observationSequence:Vector,
    scale: Boolean,
    scalingFactors: Option[Array[Double]]
  ): DenseMatrix = {
    var backwardVariables = new DenseMatrix(observationSequence.size, numberOfHiddenStates)
    if (scale)
    {
      var backwardVariablesTemp = new DenseMatrix(observationSequence.size, numberOfHiddenStates)
      // initialization
      for (index <- 0 until numberOfHiddenStates) {
        backwardVariablesTemp(observationSequence.size - 1, index) =  1;
        backwardVariables(observationSequence.size - 1, index) = scalingFactors.get(observationSequence.size - 1) * backwardVariablesTemp(observationSequence.size - 1, index)
      }

      // induction
      for (indexT <- observationSequence.size - 2 to 0 by -1) {
        for (indexN <- 0 until numberOfHiddenStates) {
	  var sum:Double = 0.0
	  for (indexM <- 0 until numberOfHiddenStates) {
            sum += backwardVariables(indexT + 1, indexM) * transitionMatrix(indexN, indexM) * emissionMatrix(indexM, observationSequence(indexT + 1).toInt) 
          }

          backwardVariablesTemp(indexT, indexN) =  sum
          backwardVariables(indexT, indexN) = backwardVariablesTemp(indexT, indexN) * scalingFactors.get(indexT)
        }
      }
    } else {
      // Initialization
      for (index <- 0 until numberOfHiddenStates) {
        backwardVariables(observationSequence.size - 1, index) = 1
      }
      // Induction
      for (indexT <- observationSequence.size - 2 to 0 by -1) {
        for (indexN <- 0 until numberOfHiddenStates) {
	  var sum:Double = 0.0
	  for (indexM <- 0 until numberOfHiddenStates) {
	  	      sum += backwardVariables(indexT + 1, indexM) * transitionMatrix(indexN, indexM) * emissionMatrix(indexM, observationSequence(indexT + 1).toInt)
	  }

          backwardVariables(indexT, indexN) = sum
	}
      }
    }

    backwardVariables
  }

  def sequenceLikelihood(forwardVariables: DenseMatrix,
    scalingFactors: Option[Array[Double]]
  ): Double = {
    var likelihood: Double = 0.0

    if (scalingFactors == None) {
      for (indexN <- 0 until forwardVariables.columnSize()) {
        likelihood += forwardVariables(forwardVariables.rowSize() - 1, indexN)
      }
    } else {
      var product: Double = 1.0
      for (indexT <- 0 until scalingFactors.get.length) {
        product = product * scalingFactors.get(indexT)
      }

      likelihood = 1.0 / product
    }

    likelihood
  }
}

class HiddenMarkovModel(val numberOfHiddenStates: Int,
  val numberOfOutputSymbols: Int,
  var transitionMatrix: Matrix = null,
  var emissionMatrix: Matrix = null,
  var initialProbabilities: Vector = null) extends UnsupervisedModel {

  var cumulativesInitialized = false
  var cumulativeTransitionMatrix:Matrix = _
  var cumulativeEmissionMatrix:Matrix = _
  var cumulativeInitialProbabilities:Vector = _
  var validateModel:Boolean = false

  if (initialProbabilities == null) {
    initialProbabilities = new DenseVector(numberOfHiddenStates)
  } else {
    validateModel = true
  }

  if (transitionMatrix == null ) {
    transitionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfHiddenStates)
  }

  if (emissionMatrix == null) {
    emissionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfOutputSymbols)
  }

  if (validateModel) {
    validate()
  }
  else {
    initModelWithRandomParameters(System.currentTimeMillis().toInt)
  }

  def getNumberOfHiddenStates: Int = {
    numberOfHiddenStates
  }

  def getNumberOfObservableSymbols: Int = {
    numberOfOutputSymbols
  }

  def getInitialProbabilities: Vector = {
    initialProbabilities
  }

  def getEmissionMatrix: Matrix = {
    emissionMatrix
  }

  def getTransitionMatrix: Matrix = {
    transitionMatrix
  }

  def getCumulativeTransitionMatrix: Matrix = {
    if (cumulativesInitialized == false) {
      cumulativeTransitionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfHiddenStates)
      for (i <- 0 until numberOfHiddenStates) {
        var sum:Double = 0;
        for (j <- 0 until numberOfHiddenStates) {
          sum = sum + transitionMatrix(i, j)
          cumulativeTransitionMatrix(i, j) = sum
        }
        cumulativeTransitionMatrix(i, numberOfHiddenStates - 1) =  1.0
      }
    }

    cumulativeTransitionMatrix
  }

  def getCumulativeEmissionMatrix: Matrix = {
    if (cumulativesInitialized == false) {
      cumulativeEmissionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfOutputSymbols)
      for (i <- 0 until numberOfHiddenStates) {
        var sum:Double = 0;
        for (j <- 0 until numberOfOutputSymbols) {
          sum = sum + emissionMatrix(i, j)
          cumulativeEmissionMatrix(i, j) = sum
        }
        cumulativeEmissionMatrix(i, numberOfOutputSymbols - 1) = 1.0
      }
    }

    cumulativeEmissionMatrix
  }

  def getCumulativeInitialProbabilities: Vector = {
    if (cumulativesInitialized == false) {
      cumulativeInitialProbabilities = new DenseVector(numberOfHiddenStates)
      var sum:Double = 0;
      for (i <- 0 until numberOfHiddenStates) {
        sum = sum + initialProbabilities(i)
        cumulativeInitialProbabilities(i) = sum
      }
      cumulativeInitialProbabilities(numberOfHiddenStates - 1) =  1.0
    }

    cumulativeInitialProbabilities
  }

  def generate(len:Int, seed:Long):(DenseVector, DenseVector) = {
    var observationSeq = new DenseVector(len)
    var hiddenSeq = new DenseVector(len)
    var rand:Random = RandomUtils.getRandom()
    if (seed != 0) {
      rand = RandomUtils.getRandom(seed)
    }
    var hiddenState:Int = 0

    var randnr:Double = rand.nextDouble()
    while (getCumulativeInitialProbabilities(hiddenState) < randnr) {
      hiddenState = hiddenState + 1
    }

    // now draw steps output states according to the cumulative
    // distributions
    for (step <- 0 until len) {
      // choose output state to given hidden state
      randnr = rand.nextDouble()
      var outputState:Int = 0
      while (getCumulativeEmissionMatrix(hiddenState, outputState) < randnr) {
        outputState = outputState + 1
      }
      observationSeq(step) = outputState
      // choose the next hidden state
      randnr = rand.nextDouble()
      var nextHiddenState:Int = 0
      while (getCumulativeTransitionMatrix(hiddenState, nextHiddenState) < randnr) {
        nextHiddenState = nextHiddenState + 1
      }
      hiddenState = nextHiddenState;
    }

    (observationSeq, hiddenSeq)
  }

  def printModel(): Unit = {
    println("Transition Matrix:")
    println(getTransitionMatrix)
    println("Emission Matrix:")
    println(getEmissionMatrix)
    println("Initial Probabilities Vector:")
    println(getInitialProbabilities)
  }

  def validate() {
    assert(numberOfHiddenStates > 0, "number of hidden states has to be greater than 0.")
    assert(numberOfOutputSymbols > 0, "number of output symbols has to be greater than 0.")
    assert(transitionMatrix.rowSize() == numberOfHiddenStates, "number of rows of transition matrix should be equal to number of hidden states")
    assert(transitionMatrix.columnSize() == numberOfHiddenStates, "number of columns of transition matrix should be equal to number of hidden states")
    assert(emissionMatrix.rowSize() == numberOfHiddenStates, "number of rows of emission matrix should be equal to number of hidden states")
    assert(emissionMatrix.columnSize() == numberOfOutputSymbols, "number of columns of emission matrix should be equal to number of output symbols")
    assert(initialProbabilities.size() == numberOfHiddenStates, "number of entries of initial probabilities vector should be equal to number of hidden states")

    var sum:Double = 0
    for (i <- 0 until initialProbabilities.size()) {
      assert(initialProbabilities(i) >= 0, "Initial probability of state is negative")
      sum += initialProbabilities(i)
    }

    assert(Math.abs(sum - 1) <= 0.000001, "Initial probabilities do not add up to 1")

    for (i <- 0 until numberOfHiddenStates) {
      sum = 0
      for (j <- 0 until numberOfHiddenStates) {
        assert(transitionMatrix(i, j) >= 0, "transition probability is negative")
        sum += transitionMatrix(i, j)
      }
      assert(Math.abs(sum - 1) <= 0.000001, "transition probabilities do not add up to 1")
    }

    for (i <- 0 until numberOfHiddenStates) {
      sum = 0
      for (j <- 0 until numberOfOutputSymbols) {
        assert(emissionMatrix(i, j) >= 0, "emission probability is negative")
        sum += emissionMatrix(i, j)
      }
      assert(Math.abs(sum - 1) <= 0.000001, "emission probabilities do not add up to 1")
    }
  }

  def initModelWithRandomParameters(seed:Long): Unit = {
    var rand:Random = RandomUtils.getRandom()
    if (seed != 0) {
      rand = RandomUtils.getRandom(seed)
    }

    // initialize the initial Probabilities
    var sum:Double = 0 
    for (i <- 0 to numberOfHiddenStates - 1) {
      val nextRand:Double = rand.nextDouble();
      initialProbabilities(i) = nextRand
      sum += nextRand;
    }
    // "normalize" the vector to generate probabilities
    for (i <- 0 to numberOfHiddenStates - 1) {
      initialProbabilities(i) = initialProbabilities(i)/sum
    }

    // initialize the transition matrix
    for (i <- 0 to numberOfHiddenStates -1) {
      sum = 0
      for (j <- 0 to numberOfHiddenStates - 1) {
        transitionMatrix(i, j) =  rand.nextDouble()
        sum += transitionMatrix(i, j)
      }
      // normalize the random values to obtain probabilities
      for (j <- 0 to numberOfHiddenStates - 1) {
        transitionMatrix(i, j) = transitionMatrix(i, j)/sum
      }
    }

    // initialize the output matrix
    for (i <- 0 to numberOfHiddenStates - 1) {
      sum = 0
      for (j <- 0 to numberOfOutputSymbols - 1) {
        emissionMatrix(i, j) = rand.nextDouble()
        sum += emissionMatrix(i, j)
      }
      // normalize the random values to obtain probabilities
      for (j <- 0 to numberOfOutputSymbols - 1) {
        emissionMatrix(i, j) = emissionMatrix(i, j)/sum
      }
    }
  }

  def likelihood(observationSequence: Vector, scale: Boolean):Double = {
    val (forwardVariables, scalingFactors) = HMMFunctions.computeForwardVariables(numberOfHiddenStates, initialProbabilities, transitionMatrix, emissionMatrix, observationSequence, scale)

    val obsLikelihood = HMMFunctions.sequenceLikelihood(forwardVariables, scalingFactors)
    obsLikelihood
  }

  def decode(observationSequence: Vector, scale: Boolean):DenseVector = {
    // probability that the most probable hidden states ends at state i at time t
    val delta = Array.ofDim[Double](observationSequence.length, numberOfHiddenStates)
    // previous hidden state in the most probable state leading up to state i at time t
    val phi = Array.ofDim[Int](observationSequence.length - 1, numberOfHiddenStates)
    var hiddenSeq = new DenseVector(observationSequence.length)

    // Initialization
    if (scale) {
      for (index <- 0 until numberOfHiddenStates) {
        delta(0)(index) = Math.log(initialProbabilities(index) * emissionMatrix(index, observationSequence(0).toInt))
      }
    } else {
      for (index <- 0 until numberOfHiddenStates) {
        delta(0)(index) = initialProbabilities(index) * emissionMatrix(index, observationSequence(0).toInt)
      }
    }

    // Induction
    // iterate over time
    if (scale) {
      for (t <- 1 until observationSequence.length) {
        // iterate over the hidden states
        for (i <- 0 until numberOfHiddenStates) {
          // find the maximum probability and most likely state
          // leading up
          // to this
          var maxState:Int = 0;
          var maxProb:Double = delta(t - 1)(0) + Math.log(transitionMatrix(0, i))
          for (j <- 1 until numberOfHiddenStates) {
            val prob:Double = delta(t - 1)(j) + Math.log(transitionMatrix(j, i))
            if (prob > maxProb) {
              maxProb = prob
              maxState = j
            }
          }
          delta(t)(i) = maxProb + Math.log(emissionMatrix(i, observationSequence(t).toInt))
          phi(t - 1)(i) = maxState
        }
      }
    } else {
      for (t <- 1 until observationSequence.length) {
        // iterate over the hidden states
        for (i <- 0 until numberOfHiddenStates) {
          // find the maximum probability and most likely state
          // leading up
          // to this
          var maxState:Int = 0
          var maxProb:Double = delta(t - 1)(0) * transitionMatrix(0, i)
          for (j <- 1 until numberOfHiddenStates) {
            val prob:Double = delta(t - 1)(j) * transitionMatrix(j, i)
            if (prob > maxProb) {
              maxProb = prob
              maxState = j
            }
          }
          delta(t)(i) = maxProb * emissionMatrix(i, observationSequence(t).toInt)
          phi(t - 1)(i) = maxState
        }
      }
    }

    // find the most likely end state for initialization
    var maxProb:Double = 0.0
    if (scale) {
      maxProb = Double.NegativeInfinity
    } else {
      maxProb = 0.0
    }

    for (i <- 0 until numberOfHiddenStates) {
      if (delta(observationSequence.length - 1)(i) > maxProb) {
        maxProb = delta(observationSequence.length - 1)(i)
        hiddenSeq(observationSequence.length - 1) = i
      }
    }

    // now backtrack to find the most likely hidden sequence
    for (t <- observationSequence.length - 2 to 0 by -1) {
      hiddenSeq(t) = phi(t)(hiddenSeq(t + 1).toInt)
    }

    hiddenSeq
  }
}

class HiddenMarkovModelFitter extends UnsupervisedFitter with Serializable {

  def setHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): (HiddenMarkovModel, Int, Double, Boolean) = {
    val initModel:HiddenMarkovModel = hyperparameters.asInstanceOf[Map[Symbol, HiddenMarkovModel]]('initModel)
    val maxNumberOfIterations:Int = hyperparameters.asInstanceOf[Map[Symbol, Int]].getOrElse('iterations, 1)
    val epsilon:Double = hyperparameters.asInstanceOf[Map[Symbol, Double]].getOrElse('epsilon, 0.1)
    val scale:Boolean = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('scale, true)
    (initModel, maxNumberOfIterations, epsilon, scale)
  }

  

  def expectedNumberOfTransitions(numberOfHiddenStates: Int,
    transitionMatrix: Matrix,
    emissionMatrix: Matrix, observationSequence:Vector, forwardVariables: DenseMatrix,
    backwardVariables: DenseMatrix, likelihood: Double, indexN: Int, indexM: Int, scale: Boolean): Double = {
    var numTransitions:Double = 0.0
    for (indexT <- 0 until observationSequence.size - 1) {
      numTransitions += forwardVariables(indexT, indexN) * emissionMatrix(indexM, observationSequence(indexT + 1).toInt) * backwardVariables(indexT + 1, indexM)
    }

    if (scale) {
      numTransitions = (numTransitions * transitionMatrix(indexN, indexM))
    } else {
      numTransitions = (numTransitions * transitionMatrix(indexN, indexM)) / likelihood
    }

    numTransitions
  }

  def expectedNumberOfEmissions(observationSequence:Vector, forwardVariables: DenseMatrix,
    backwardVariables: DenseMatrix, likelihood: Double, indexN: Int, indexM: Int, scale: Boolean, scalingFactors: Option[Array[Double]]): Double = {
    var numEmissions:Double = 0.0
    for (indexT <- 0 until observationSequence.size) {
      if (observationSequence(indexT).toInt == indexM) {
        if (scale) {
          numEmissions += forwardVariables(indexT, indexN) * backwardVariables(indexT, indexN)/scalingFactors.get(indexT)
        } else {
          numEmissions += forwardVariables(indexT, indexN) * backwardVariables(indexT, indexN)
        }
      }
    }

    if (scale == false) {
      numEmissions = numEmissions / likelihood
    }

    numEmissions
  }

  def checkForConvergence(model: HiddenMarkovModel, nextModel:HiddenMarkovModel , epsilon: Double): Boolean = {
    // check convergence of transitionProbabilities
    var diff: Double = 0.0;
    for (indexN <- 0 until model.getNumberOfHiddenStates) {
      for (indexM <- 0 until model.getNumberOfHiddenStates) {
        val oldVal:Double = model.getTransitionMatrix(indexN, indexM)
        val newVal:Double = nextModel.getTransitionMatrix(indexN, indexM)
        val tmp: Double = oldVal - newVal
        diff += tmp * tmp
      }
    }

    var norm: Double = Math.sqrt(diff)
    diff = 0
    // check convergence of emissionProbabilities
    for (indexN <- 0 until model.getNumberOfHiddenStates) {
      for (indexM <- 0 until model.getNumberOfObservableSymbols) {
        val oldVal:Double = model.getEmissionMatrix(indexN, indexM)
        val newVal:Double = nextModel.getEmissionMatrix(indexN, indexM)
        val tmp: Double = oldVal - newVal
        diff += tmp * tmp
      }
    }

    norm += Math.sqrt(diff)
    // iteration has converged

    norm < epsilon
  }

  def fit[K](observations: DrmLike[K],
    hyperparameters: (Symbol, Any)*): HiddenMarkovModel = {
    
    val (initModel, maxNumberOfIterations, epsilon, scale) = setHyperparameters(hyperparameters.toMap)
    implicit val ctx = observations.context
    implicit val ktag =  observations.keyClassTag

    var curModel:HiddenMarkovModel = initModel
    var iter = 0
    var stop = false

    while ((iter < maxNumberOfIterations) && (!stop)) {
      iter = iter + 1
      var transitionMatrix = new DenseMatrix(curModel.getNumberOfHiddenStates, curModel.getNumberOfHiddenStates)
      var emissionMatrix = new DenseMatrix(curModel.getNumberOfHiddenStates, curModel.getNumberOfObservableSymbols)
      var initialProbabilities = new DenseVector(curModel.getNumberOfHiddenStates)

      // Broadcast the current model parameters
      val bcastInitialProbabilities = drmBroadcast(curModel.getInitialProbabilities)
      val bcastTransitionMatrix = drmBroadcast(curModel.getTransitionMatrix)
      val bcastEmissionMatrix = drmBroadcast(curModel.getEmissionMatrix)
      val numberOfHiddenStates = curModel.getNumberOfHiddenStates
      val numberOfObservableSymbols = curModel.getNumberOfObservableSymbols

      var numCols = numberOfHiddenStates + numberOfHiddenStates * numberOfHiddenStates + numberOfHiddenStates * numberOfObservableSymbols

      val resultDrm = observations.mapBlock(numCols) {
        case (keys, block:Matrix) => {
          val outputMatrix = new DenseMatrix(block.nrow, numCols)

          for (obsIndex <- 0 until block.nrow) {
            val observation = block.viewRow(obsIndex)
            
            val (forwardVariables, scalingFactors) = HMMFunctions.computeForwardVariables(numberOfHiddenStates, bcastInitialProbabilities, bcastTransitionMatrix, bcastEmissionMatrix, observation, scale)
            val backwardVariables = HMMFunctions.computeBackwardVariables(numberOfHiddenStates, bcastInitialProbabilities, bcastTransitionMatrix, bcastEmissionMatrix, observation, scale, scalingFactors)
            val obsLikelihood = HMMFunctions.sequenceLikelihood(forwardVariables, scalingFactors)
      
            // recompute initial probabilities
            if (scale) {
              for (index <- 0 until numberOfHiddenStates) {
                val prob:Double = forwardVariables(0, index) * backwardVariables(0, index) / scalingFactors.get(0)
                outputMatrix(obsIndex, index) =  prob
              }
            } else {
              for (index <- 0 until numberOfHiddenStates) {
                val prob:Double = forwardVariables(0, index) * backwardVariables(0, index) / obsLikelihood
                outputMatrix(obsIndex, index) = prob
              }
            }

            // recompute transitionmatrix
            for (indexN <- 0 until numberOfHiddenStates) {
              for (indexM <- 0 until numberOfHiddenStates) {
                val numTransitions = expectedNumberOfTransitions(numberOfHiddenStates, bcastTransitionMatrix, bcastEmissionMatrix, observation, forwardVariables, backwardVariables, obsLikelihood, indexN, indexM, scale)
                outputMatrix(obsIndex, numberOfHiddenStates * (indexN + 1) + indexM) = numTransitions
              }
            }

            // recompute emissionmatrix
            for (indexN <- 0 until numberOfHiddenStates) {
              var denominator:Double = 0.0
              for (indexM <- 0 until numberOfObservableSymbols) {
                var numEmissions:Double = expectedNumberOfEmissions(observation, forwardVariables, backwardVariables, obsLikelihood, indexN, indexM, scale, scalingFactors)
                outputMatrix(obsIndex, numberOfHiddenStates + numberOfHiddenStates * numberOfHiddenStates + numberOfObservableSymbols * indexN + indexM) = numEmissions
              }
            }
          }

          (keys, outputMatrix)
        }
      }

      val countsMatrix = resultDrm.collect

      for (indexN <- 0 until numberOfHiddenStates) {
        initialProbabilities(indexN) = 0.0
        for (indexM <- 0 until numberOfHiddenStates) {
          transitionMatrix(indexN, indexM) = 0.0
        }
        
        for (indexM <- 0 until numberOfObservableSymbols) {
          emissionMatrix(indexN, indexM) = 0.0          
        }
      }

      for (index <- 0 until countsMatrix.nrow) {
        for (indexM <- 0 until numberOfHiddenStates) {
          initialProbabilities(indexM) = initialProbabilities(indexM) + countsMatrix(index, indexM)
        }

        for (indexN <- 0 until numberOfHiddenStates) {
          for (indexM <- 0 until numberOfHiddenStates) {
            transitionMatrix(indexN, indexM) =  transitionMatrix(indexN, indexM) + countsMatrix(index, numberOfHiddenStates * (indexN + 1) + indexM)
          }
        }

        for (indexN <- 0 until numberOfHiddenStates) {
          for (indexM <- 0 until numberOfObservableSymbols) {
            emissionMatrix(indexN, indexM) = emissionMatrix(indexN, indexM) + countsMatrix(index, numberOfHiddenStates + numberOfHiddenStates * numberOfHiddenStates + numberOfObservableSymbols * indexN + indexM)
          }
        }
      }

      // normalize the matrices
      var totalI:Double = 0.0
      for (indexN <- 0 until numberOfHiddenStates) {
        totalI += initialProbabilities(indexN)

        var total:Double = 0.0
        for (indexM <- 0 until numberOfHiddenStates) {
          total += transitionMatrix(indexN, indexM)
        }

        for (indexM <- 0 until numberOfHiddenStates) {
          transitionMatrix(indexN, indexM) = transitionMatrix(indexN, indexM)/total
        }

        total = 0.0
        for (indexM <- 0 until numberOfObservableSymbols) {
          total += emissionMatrix(indexN, indexM)
        }

        for (indexM <- 0 until numberOfObservableSymbols) {
          emissionMatrix(indexN, indexM) = emissionMatrix(indexN, indexM)/total
        }
      }

      for (indexN <- 0 until numberOfHiddenStates) {
        initialProbabilities(indexN) = initialProbabilities(indexN)/totalI
      }

      val newModel:HiddenMarkovModel = new HiddenMarkovModel(numberOfHiddenStates, numberOfObservableSymbols, transitionMatrix, emissionMatrix, initialProbabilities)

      if (checkForConvergence(curModel, newModel, epsilon)) {
        stop = true
      }

      curModel = newModel
    }

    curModel
  }

  // used to store the model if `fitTransform` method called
  var model: HiddenMarkovModel = _
}
