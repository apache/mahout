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

package org.apache.mahout.classifier.sequencelearning.hmm

import org.apache.mahout.classifier.stats.{ResultAnalyzer, ClassifierResult}
import org.apache.mahout.math._
import scalabindings._
import scalabindings.RLikeOps._
import drm.RLikeDrmOps._
import drm._
import scala.reflect.ClassTag
import scala.language.asInstanceOf
import collection._
import scala.collection.JavaConversions._
import scala.util.Random
import org.apache.mahout.common.RandomUtils

trait HiddenMarkovModel extends java.io.Serializable {
  
  def computeForwardVariables(numberOfHiddenStates: Int,
    initialProbabilities: Vector,
    transitionMatrix: Matrix,
    emissionMatrix: Matrix,
    observationSequence:Vector,
    scale: Boolean
  ): (DenseMatrix, Option[Array[Double]]) = {
    var forwardVariables = new DenseMatrix(observationSequence.length, numberOfHiddenStates)
    var scalingFactors = None: Option[Array[Double]]

    if (scale) {
      scalingFactors = Some(new Array(observationSequence.length))
      var forwardVariablesTemp = new DenseMatrix(observationSequence.length, numberOfHiddenStates)
      // Initialization
      for (index <- 0 until numberOfHiddenStates) {
        forwardVariablesTemp.setQuick(0, index, initialProbabilities.getQuick(index)
	    			* emissionMatrix.getQuick(index, observationSequence(0).toInt));
      }

      var sum:Double = 0.0
      for (index <- 0 until numberOfHiddenStates) {
        sum += forwardVariablesTemp.getQuick(0, index)
      }

      scalingFactors.get(0) = 1.0/sum

      for (index <- 0 until numberOfHiddenStates) {
        forwardVariables.setQuick(0, index, forwardVariablesTemp.getQuick(0, index) * scalingFactors.get(0))
      }

      // Induction
      for (indexT <- 1 until observationSequence.length) {
        for (indexN <- 0 until numberOfHiddenStates) {
	  var sumA:Double = 0.0
	  for (indexM <- 0 until numberOfHiddenStates) {
            sumA += forwardVariables.getQuick(indexT - 1, indexM) * transitionMatrix.getQuick(indexM, indexN) * emissionMatrix.getQuick(indexN, observationSequence(indexT).toInt)
          }

          forwardVariablesTemp.setQuick(indexT, indexN, sumA)
        }

        var sumT:Double = 0.0
        for (indexN <- 0 until numberOfHiddenStates) {
          sumT += forwardVariablesTemp.getQuick(indexT, indexN)
        }

        scalingFactors.get(indexT) = 1.0/sumT

        for (indexN <- 0 until numberOfHiddenStates) {
          forwardVariables.setQuick(indexT, indexN, scalingFactors.get(indexT) * forwardVariablesTemp.getQuick(indexT, indexN))
        }
      }
    } else {
      // Initialization
      for (index <- 0 until numberOfHiddenStates) {
        forwardVariables.setQuick(0, index, initialProbabilities.getQuick(index)
	    			* emissionMatrix.getQuick(index, observationSequence(0).toInt));
      }

      // Induction
      for (indexT <- 1 until observationSequence.length) {
        for (indexN <- 0 until numberOfHiddenStates) {
	  var sum:Double = 0.0
	  for (indexM <- 0 until numberOfHiddenStates) {
	    sum += forwardVariables.getQuick(indexT - 1, indexM) * transitionMatrix.getQuick(indexM, indexN);
	  }

	  forwardVariables.setQuick(indexT, indexN, sum * emissionMatrix.getQuick(indexN, observationSequence(indexT).toInt))
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
    var backwardVariables = new DenseMatrix(observationSequence.length, numberOfHiddenStates)
    if (scale)
    {
      var backwardVariablesTemp = new DenseMatrix(observationSequence.length, numberOfHiddenStates)
      // initialization
      for (index <- 0 until numberOfHiddenStates) {
        backwardVariablesTemp.setQuick(observationSequence.length - 1, index, 1);
        backwardVariables.setQuick(observationSequence.length - 1, index, scalingFactors.get(observationSequence.length - 1) * backwardVariablesTemp.getQuick(observationSequence.length - 1, index))
      }

      // induction
      for (indexT <- observationSequence.length - 2 to 0 by -1) {
        for (indexN <- 0 until numberOfHiddenStates) {
	  var sum:Double = 0.0
	  for (indexM <- 0 until numberOfHiddenStates) {
            sum += backwardVariables.getQuick(indexT + 1, indexM) * transitionMatrix.getQuick(indexN, indexM) * emissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt) 
          }

          backwardVariablesTemp.setQuick(indexT, indexN, sum)
          backwardVariables.setQuick(indexT, indexN, backwardVariablesTemp.getQuick(indexT, indexN) * scalingFactors.get(indexT))
        }
      }
    } else {
      // Initialization
      for (index <- 0 until numberOfHiddenStates) {
        backwardVariables.setQuick(observationSequence.length - 1, index, 1)
      }
      // Induction
      for (indexT <- observationSequence.length - 2 to 0 by -1) {
        for (indexN <- 0 until numberOfHiddenStates) {
	  var sum:Double = 0.0
	  for (indexM <- 0 until numberOfHiddenStates) {
	  	      sum += backwardVariables.getQuick(indexT + 1, indexM) * transitionMatrix.getQuick(indexN, indexM) * emissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt)
	  }

          backwardVariables.setQuick(indexT, indexN, sum)
	}
      }
    }

    backwardVariables
  }

  def expectedNumberOfTransitions(numberOfHiddenStates: Int,
    transitionMatrix: Matrix,
    emissionMatrix: Matrix, observationSequence:Vector, forwardVariables: DenseMatrix,
    backwardVariables: DenseMatrix, likelihood: Double, indexN: Int, indexM: Int, scale: Boolean): Double = {
    var numTransitions:Double = 0.0
    for (indexT <- 0 until observationSequence.length - 1) {
      numTransitions += forwardVariables.getQuick(indexT, indexN) * emissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt) * backwardVariables.getQuick(indexT + 1, indexM)
    }

    if (scale) {
      numTransitions = (numTransitions * transitionMatrix.getQuick(indexN, indexM))
    } else {
      numTransitions = (numTransitions * transitionMatrix.getQuick(indexN, indexM)) / likelihood
    }

    numTransitions
  }

  def expectedNumberOfEmissions(observationSequence:Vector, forwardVariables: DenseMatrix,
    backwardVariables: DenseMatrix, likelihood: Double, indexN: Int, indexM: Int, scale: Boolean, scalingFactors: Option[Array[Double]]): Double = {
    var numEmissions:Double = 0.0
    for (indexT <- 0 until observationSequence.length) {
      if (observationSequence(indexT).toInt == indexM) {
        if (scale) {
          numEmissions += forwardVariables.getQuick(indexT, indexN) * backwardVariables.getQuick(indexT, indexN)/scalingFactors.get(indexT)
        } else {
          numEmissions += forwardVariables.getQuick(indexT, indexN) * backwardVariables.getQuick(indexT, indexN)
        }
      }
    }

    if (scale == false) {
      numEmissions = numEmissions / likelihood
    }

    numEmissions
  }

  def sequenceLikelihood(forwardVariables: DenseMatrix,
    scalingFactors: Option[Array[Double]]
  ): Double = {
    var likelihood: Double = 0.0

    if (scalingFactors == None) {
      for (indexN <- 0 until forwardVariables.columnSize()) {
        likelihood += forwardVariables.getQuick(forwardVariables.rowSize() - 1, indexN)
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

  def checkForConvergence(model: HMMModel, nextModel: HMMModel, epsilon: Double): Boolean = {
    // check convergence of transitionProbabilities
    var diff: Double = 0.0;
    for (indexN <- 0 until model.getNumberOfHiddenStates) {
      for (indexM <- 0 until model.getNumberOfHiddenStates) {
        val oldVal:Double = model.getTransitionMatrix.getQuick(indexN, indexM)
        val newVal:Double = nextModel.getTransitionMatrix.getQuick(indexN, indexM)
        val tmp: Double = oldVal - newVal
        diff += tmp * tmp
      }
    }

    var norm: Double = Math.sqrt(diff)
    diff = 0
    // check convergence of emissionProbabilities
    for (indexN <- 0 until model.getNumberOfHiddenStates) {
      for (indexM <- 0 until model.getNumberOfObservableSymbols) {
        val oldVal:Double = model.getEmissionMatrix.getQuick(indexN, indexM)
        val newVal:Double = nextModel.getEmissionMatrix.getQuick(indexN, indexM)
        val tmp: Double = oldVal - newVal
        diff += tmp * tmp
      }
    }

    norm += Math.sqrt(diff)
    // iteration has converged

    norm < epsilon
  }

  def train(initModel: HMMModel,
    observations: DrmLike[Long],
    epsilon: Double,
    maxNumberOfIterations:Int,
    scale: Boolean = false
  )(implicit ctx: DistributedContext): HMMModel = {
    var curModel:HMMModel = initModel
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
            val (forwardVariables, scalingFactors) = computeForwardVariables(numberOfHiddenStates, bcastInitialProbabilities, bcastTransitionMatrix, bcastEmissionMatrix, observation, scale)
            val backwardVariables = computeBackwardVariables(numberOfHiddenStates, bcastInitialProbabilities, bcastTransitionMatrix, bcastEmissionMatrix, observation, scale, scalingFactors)
            val obsLikelihood = sequenceLikelihood(forwardVariables, scalingFactors)
      
            // recompute initial probabilities
            if (scale) {
              for (index <- 0 until numberOfHiddenStates) {
                val prob:Double = forwardVariables.getQuick(0, index) * backwardVariables.getQuick(0, index) / scalingFactors.get(0)
                outputMatrix.setQuick(obsIndex, index, prob)
              }
            } else {
              for (index <- 0 until numberOfHiddenStates) {
                val prob:Double = forwardVariables.getQuick(0, index) * backwardVariables.getQuick(0, index) / obsLikelihood
                outputMatrix.setQuick(obsIndex, index, prob)
              }
            }

            // recompute transitionmatrix
            for (indexN <- 0 until numberOfHiddenStates) {
              for (indexM <- 0 until numberOfHiddenStates) {
                val numTransitions = expectedNumberOfTransitions(numberOfHiddenStates, bcastTransitionMatrix, bcastEmissionMatrix, observation, forwardVariables, backwardVariables, obsLikelihood, indexN, indexM, scale)
                outputMatrix.setQuick(obsIndex, numberOfHiddenStates * (indexN + 1) + indexM, numTransitions)
              }
            }

            // recompute emissionmatrix
            for (indexN <- 0 until numberOfHiddenStates) {
              var denominator:Double = 0.0
              for (indexM <- 0 until numberOfObservableSymbols) {
                var numEmissions:Double = expectedNumberOfEmissions(observation, forwardVariables, backwardVariables, obsLikelihood, indexN, indexM, scale, scalingFactors)
                outputMatrix.setQuick(obsIndex, numberOfHiddenStates + numberOfHiddenStates * numberOfHiddenStates + numberOfObservableSymbols * indexN + indexM, numEmissions)
              }
            }
          }

          (keys, outputMatrix)
        }
      }

      val countsMatrix = resultDrm.collect

      for (indexN <- 0 until numberOfHiddenStates) {
        initialProbabilities.setQuick(indexN, 0.0)
        for (indexM <- 0 until numberOfHiddenStates) {
          transitionMatrix.setQuick(indexN, indexM, 0.0)
        }
        
        for (indexM <- 0 until numberOfObservableSymbols) {
          emissionMatrix.setQuick(indexN, indexM, 0.0)          
        }
      }

      for (index <- 0 until countsMatrix.nrow) {
        for (indexM <- 0 until numberOfHiddenStates) {
          initialProbabilities.setQuick(indexM, initialProbabilities.getQuick(indexM) + countsMatrix.getQuick(index, indexM))
        }

        for (indexN <- 0 until numberOfHiddenStates) {
          for (indexM <- 0 until numberOfHiddenStates) {
            transitionMatrix.setQuick(indexN, indexM, transitionMatrix.getQuick(indexN, indexM) + countsMatrix.getQuick(index, numberOfHiddenStates * (indexN + 1) + indexM))
          }
        }

        for (indexN <- 0 until numberOfHiddenStates) {
          for (indexM <- 0 until numberOfObservableSymbols) {
            emissionMatrix.setQuick(indexN, indexM, emissionMatrix.getQuick(indexN, indexM) + countsMatrix.getQuick(index, numberOfHiddenStates + numberOfHiddenStates * numberOfHiddenStates + numberOfObservableSymbols * indexN + indexM))
          }
        }
      }

      // normalize the matrices
      var totalI:Double = 0.0
      for (indexN <- 0 until numberOfHiddenStates) {
        totalI += initialProbabilities.getQuick(indexN)

        var total:Double = 0.0
        for (indexM <- 0 until numberOfHiddenStates) {
          total += transitionMatrix.getQuick(indexN, indexM)
        }

        for (indexM <- 0 until numberOfHiddenStates) {
          transitionMatrix.setQuick(indexN, indexM, transitionMatrix.getQuick(indexN, indexM)/total)
        }

        total = 0.0
        for (indexM <- 0 until numberOfObservableSymbols) {
          total += emissionMatrix.getQuick(indexN, indexM)
        }

        for (indexM <- 0 until numberOfObservableSymbols) {
          emissionMatrix.setQuick(indexN, indexM, emissionMatrix.getQuick(indexN, indexM)/total)
        }
      }

      for (indexN <- 0 until numberOfHiddenStates) {
        initialProbabilities.setQuick(indexN, initialProbabilities.getQuick(indexN)/totalI)
      }

      val newModel:HMMModel = new HMMModel(numberOfHiddenStates, numberOfObservableSymbols, transitionMatrix, emissionMatrix, initialProbabilities)

      if (checkForConvergence(curModel, newModel, epsilon)) {
        stop = true
      }

      curModel = newModel
    }

    curModel
  }
 
  def likelihood(model: HMMModel, observationSequence: Vector, scale: Boolean):Double = {
    val (forwardVariables, scalingFactors) = computeForwardVariables(model.getNumberOfHiddenStates, model.getInitialProbabilities, model.getTransitionMatrix, model.getEmissionMatrix, observationSequence, scale)

    val obsLikelihood = sequenceLikelihood(forwardVariables, scalingFactors)
    obsLikelihood
  }

  def decode(model: HMMModel, observationSequence: Vector, scale: Boolean):DenseVector = {
    // probability that the most probable hidden states ends at state i at time t
    val delta = Array.ofDim[Double](observationSequence.length, model.getNumberOfHiddenStates)
    // previous hidden state in the most probable state leading up to state i at time t
    val phi = Array.ofDim[Int](observationSequence.length - 1, model.getNumberOfHiddenStates)
    var hiddenSeq = new DenseVector(observationSequence.length)

    val initialProbabilities:Vector = model.getInitialProbabilities
    val emissionMatrix:Matrix = model.getEmissionMatrix
    val transitionMatrix:Matrix = model.getTransitionMatrix

    // Initialization
    if (scale) {
      for (index <- 0 until model.getNumberOfHiddenStates) {
        delta(0)(index) = Math.log(initialProbabilities.getQuick(index) * emissionMatrix.getQuick(index, observationSequence(0).toInt))
      }
    } else {
      for (index <- 0 until model.getNumberOfHiddenStates) {
        delta(0)(index) = initialProbabilities.getQuick(index) * emissionMatrix.getQuick(index, observationSequence(0).toInt)
      }
    }

    // Induction
    // iterate over time
    if (scale) {
      for (t <- 1 until observationSequence.length) {
        // iterate over the hidden states
        for (i <- 0 until model.getNumberOfHiddenStates) {
          // find the maximum probability and most likely state
          // leading up
          // to this
          var maxState:Int = 0;
          var maxProb:Double = delta(t - 1)(0) + Math.log(transitionMatrix.getQuick(0, i))
          for (j <- 1 until model.getNumberOfHiddenStates) {
            val prob:Double = delta(t - 1)(j) + Math.log(transitionMatrix.getQuick(j, i))
            if (prob > maxProb) {
              maxProb = prob
              maxState = j
            }
          }
          delta(t)(i) = maxProb + Math.log(emissionMatrix.getQuick(i, observationSequence(t).toInt))
          phi(t - 1)(i) = maxState
        }
      }
    } else {
      for (t <- 1 until observationSequence.length) {
        // iterate over the hidden states
        for (i <- 0 until model.getNumberOfHiddenStates) {
          // find the maximum probability and most likely state
          // leading up
          // to this
          var maxState:Int = 0
          var maxProb:Double = delta(t - 1)(0) * transitionMatrix.getQuick(0, i)
          for (j <- 1 until model.getNumberOfHiddenStates) {
            val prob:Double = delta(t - 1)(j) * transitionMatrix.getQuick(j, i)
            if (prob > maxProb) {
              maxProb = prob
              maxState = j
            }
          }
          delta(t)(i) = maxProb * emissionMatrix.getQuick(i, observationSequence(t).toInt)
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

    for (i <- 0 until model.getNumberOfHiddenStates) {
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

  def generate(model: HMMModel, len:Int, seed:Long):(DenseVector, DenseVector) = {
    var observationSeq = new DenseVector(len)
    var hiddenSeq = new DenseVector(len)
    var rand:Random = RandomUtils.getRandom()
    if (seed != 0) {
      rand = RandomUtils.getRandom(seed)
    }
    var hiddenState:Int = 0

    var randnr:Double = rand.nextDouble()
    while (model.getCumulativeInitialProbabilities.getQuick(hiddenState) < randnr) {
      hiddenState = hiddenState + 1
    }

    // now draw steps output states according to the cumulative
    // distributions
    for (step <- 0 until len) {
      // choose output state to given hidden state
      randnr = rand.nextDouble()
      var outputState:Int = 0
      while (model.getCumulativeEmissionMatrix.getQuick(hiddenState, outputState) < randnr) {
        outputState = outputState + 1
      }
      observationSeq.setQuick(step, outputState)
      // choose the next hidden state
      randnr = rand.nextDouble()
      var nextHiddenState:Int = 0
      while (model.getCumulativeTransitionMatrix.getQuick(hiddenState, nextHiddenState) < randnr) {
        nextHiddenState = nextHiddenState + 1
      }
      hiddenState = nextHiddenState;
    }

    (observationSeq, hiddenSeq)
  }
}

object HiddenMarkovModel extends HiddenMarkovModel with java.io.Serializable

