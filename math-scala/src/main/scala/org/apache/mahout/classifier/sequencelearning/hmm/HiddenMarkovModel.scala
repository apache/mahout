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


trait HiddenMarkovModel extends java.io.Serializable {
  
  def computeForwardVariables(initModel: HMMModel,
    observationSequence:Vector,
    scale: Boolean
  ): (DenseMatrix, Option[Array[Double]]) = {
    var forwardVariables = new DenseMatrix(observationSequence.length, initModel.getNumberOfHiddenStates)
    var scalingFactors = None: Option[Array[Double]]

    if (scale) {
      scalingFactors = Some(new Array(observationSequence.length));
      var forwardVariablesTemp = new DenseMatrix(observationSequence.length, initModel.getNumberOfHiddenStates)
      // Initialization
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        forwardVariablesTemp.setQuick(0, index, initModel.getInitialProbabilities.getQuick(index)
	    			* initModel.getEmissionMatrix.getQuick(index, observationSequence(0).toInt));
      }

      var sum:Double = 0.0
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        sum += forwardVariablesTemp.getQuick(0, index)
      }

      scalingFactors.get(0) = 1.0/sum

      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        forwardVariables.setQuick(0, index, forwardVariablesTemp.getQuick(0, index) * scalingFactors.get(0))
      }

      // Induction
      for (indexT <- 1 to observationSequence.length - 1) {
        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
	  var sumA:Double = 0.0
	  for (indexM <- 0 to initModel.getNumberOfHiddenStates - 1) {
            sumA += forwardVariables.getQuick(indexT - 1, indexM) * initModel.getTransitionMatrix.getQuick(indexM, indexN) * initModel.getEmissionMatrix.getQuick(indexN, observationSequence(indexT).toInt)
          }

          forwardVariablesTemp.setQuick(indexT, indexN, sumA)
        }

        var sumT:Double = 0.0
        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
          sumT += forwardVariablesTemp.getQuick(indexT, indexN)
        }

        scalingFactors.get(indexT) = 1.0/sumT

        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
          forwardVariables.setQuick(indexT, indexN, scalingFactors.get(indexT) * forwardVariablesTemp.getQuick(indexT, indexN))
        }
      }
    } else {
      // Initialization
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        forwardVariables.setQuick(0, index, initModel.getInitialProbabilities.getQuick(index)
	    			* initModel.getEmissionMatrix.getQuick(index, observationSequence(0).toInt));
      }

      // Induction
      for (indexT <- 1 to observationSequence.length - 1) {
        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
	  var sum:Double = 0.0
	  for (indexM <- 0 to initModel.getNumberOfHiddenStates - 1) {
	    sum += forwardVariables.getQuick(indexT - 1, indexM) * initModel.getTransitionMatrix.getQuick(indexM, indexN);
	  }

	  forwardVariables.setQuick(indexT, indexN, sum * initModel.getEmissionMatrix.getQuick(indexN, observationSequence(indexT).toInt))
	}
      }
    }

    (forwardVariables, scalingFactors)
  }

  def computeBackwardVariables(initModel: HMMModel,
    observationSequence:Vector,
    scale: Boolean,
    scalingFactors: Option[Array[Double]]
  ): DenseMatrix = {
    var backwardVariables = new DenseMatrix(observationSequence.length, initModel.getNumberOfHiddenStates)
    if (scale)
    {
      var backwardVariablesTemp = new DenseMatrix(observationSequence.length, initModel.getNumberOfHiddenStates)
      // initialization
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        backwardVariablesTemp.setQuick(observationSequence.length - 1, index, 1);
        backwardVariables.setQuick(observationSequence.length - 1, index, scalingFactors.get(observationSequence.length - 1) * backwardVariablesTemp.getQuick(observationSequence.length - 1, index))
      }

      // induction
      for (indexT <- observationSequence.length - 2 to 0 by -1) {
        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
	  var sum:Double = 0.0
	  for (indexM <- 0 to initModel.getNumberOfHiddenStates - 1) {
            sum += backwardVariables.getQuick(indexT + 1, indexM) * initModel.getTransitionMatrix.getQuick(indexN, indexM) * initModel.getEmissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt) 
          }

          backwardVariablesTemp.setQuick(indexT, indexN, sum)
          backwardVariables.setQuick(indexT, indexN, backwardVariablesTemp.getQuick(indexT, indexN) * scalingFactors.get(indexT))
        }
      }
    } else {
      // Initialization
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        backwardVariables.setQuick(observationSequence.length - 1, index, 1)
      }
      // Induction
      for (indexT <- observationSequence.length - 2 to 0 by -1) {
        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
	  var sum:Double = 0.0
	  for (indexM <- 0 to initModel.getNumberOfHiddenStates - 1) {
	  	      sum += backwardVariables.getQuick(indexT + 1, indexM) * initModel.getTransitionMatrix.getQuick(indexN, indexM) * initModel.getEmissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt)
	  }

          backwardVariables.setQuick(indexT, indexN, sum)
	}
      }
    }

    backwardVariables
  }

  def expectedNumberOfTransitions(model: HMMModel, observationSequence:Vector, forwardVariables: DenseMatrix,
    backwardVariables: DenseMatrix, likelihood: Double, indexN: Int, indexM: Int): Double = {
    var numTransitions:Double = 0.0
    for (indexT <- 0 to observationSequence.length - 2) {
      numTransitions += forwardVariables.getQuick(indexT, indexN) * model.getTransitionMatrix.getQuick(indexN, indexM) * model.getEmissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt) * backwardVariables.getQuick(indexT + 1, indexM)
    }

    numTransitions = numTransitions / likelihood
    numTransitions
  }

  def expectedNumberOfEmissions(observationSequence:Vector, forwardVariables: DenseMatrix,
    backwardVariables: DenseMatrix, likelihood: Double, indexN: Int, indexM: Int): Double = {
    var numEmissions:Double = 0.0
    for (indexT <- 0 to observationSequence.length - 1) {
      if (observationSequence(indexT).toInt == indexM) {
        numEmissions += forwardVariables.getQuick(indexT, indexN) * backwardVariables.getQuick(indexT, indexN)
      }
    }

    numEmissions = numEmissions / likelihood
    numEmissions
  }

  def sequenceLikelihood(forwardVariables: DenseMatrix,
    scalingFactors: Option[Array[Double]]
  ): Double = {
    var likelihood: Double = 0.0

    if (scalingFactors == None) {
      for (indexN <- 0 to forwardVariables.columnSize() - 1) {
        likelihood += forwardVariables.getQuick(forwardVariables.rowSize() - 1, indexN)
      }
    } else {
      var product: Double = 1.0
      for (indexT <- 0 to scalingFactors.get.length - 1) {
        product = product * scalingFactors.get(indexT)
      }

      likelihood = 1.0 / product
    }

    likelihood
  }

  def checkForConvergence(model: HMMModel, nextModel: HMMModel, epsilon: Double): Boolean = {
    // check convergence of transitionProbabilities
    var diff: Double = 0.0;
    for (indexN <- 0 to model.getNumberOfHiddenStates - 1) {
      for (indexM <- 0 to model.getNumberOfHiddenStates - 1) {
        val tmp: Double = model.getTransitionMatrix.getQuick(indexN, indexM)
        - nextModel.getTransitionMatrix.getQuick(indexN, indexM)
        diff += tmp * tmp
      }
    }

    var norm: Double = Math.sqrt(diff)
    diff = 0
    // check convergence of emissionProbabilities
    for (indexN <- 0 to model.getNumberOfHiddenStates - 1) {
      for (indexM <- 0 to model.getNumberOfObservableSymbols - 1) {
        val tmp: Double = model.getEmissionMatrix.getQuick(indexN, indexM)
        - nextModel.getEmissionMatrix.getQuick(indexN, indexM)
        diff += tmp * tmp
      }
    }

    norm += Math.sqrt(diff)
    // iteration has converged
    (norm < epsilon)
  }

  def train(initModel: HMMModel,
    observations: DrmLike[Long],
    epsilon: Double,
    maxNumberOfIterations:Int,
    scale: Boolean = false
  ): HMMModel = {
    var curModel:HMMModel = initModel
    var iter = 0
    var stop = false
    while ((iter < maxNumberOfIterations) && (!stop)) {
      iter = iter + 1
      var transitionMatrix = new DenseMatrix(curModel.getNumberOfHiddenStates, curModel.getNumberOfHiddenStates)
      var emissionMatrix = new DenseMatrix(curModel.getNumberOfHiddenStates, curModel.getNumberOfObservableSymbols)
      var initialProbabilities = new DenseVector(curModel.getNumberOfHiddenStates)

      val observationsMatrix = observations.collect
      val observation = observationsMatrix.viewRow(0)
      val (forwardVariables, scalingFactors) = computeForwardVariables(curModel, observation, scale)
      val backwardVariables = computeBackwardVariables(curModel, observation, scale, scalingFactors)

      if (scale) {

      } else {
        val obsLikelihood = sequenceLikelihood(forwardVariables, None)
        // recompute initial probabilities
        for (index <- 0 to curModel.getNumberOfHiddenStates - 1) {
          initialProbabilities.setQuick(index, forwardVariables.getQuick(0, index) * backwardVariables.getQuick(0, index) / obsLikelihood)
        }

        // recompute transitionmatrix
        for (indexN <- 0 to curModel.getNumberOfHiddenStates - 1) {
          var denominator:Double = 0.0
          for (indexM <- 0 to curModel.getNumberOfHiddenStates - 1) {
            val numTransitions = expectedNumberOfTransitions(curModel, observation, forwardVariables, backwardVariables, obsLikelihood, indexN, indexM)
            denominator += numTransitions
            transitionMatrix.setQuick(indexN, indexM, numTransitions)
          }

          for (indexM <- 0 to curModel.getNumberOfHiddenStates - 1) {
            transitionMatrix.setQuick(indexN, indexM, transitionMatrix.getQuick(indexN, indexM)/denominator)
          }
        }

        // recompute emissionmatrix
        for (indexN <- 0 to curModel.getNumberOfHiddenStates - 1) {
          var denominator:Double = 0.0
          for (indexM <- 0 to curModel.getNumberOfObservableSymbols - 1) {
            var numEmissions:Double = expectedNumberOfEmissions(observation, forwardVariables, backwardVariables, obsLikelihood, indexN, indexM)
            denominator += numEmissions
            emissionMatrix.setQuick(indexN, indexM, numEmissions)
          }

          for (indexM <- 0 to curModel.getNumberOfObservableSymbols - 1) {
            emissionMatrix.setQuick(indexN, indexM, emissionMatrix.getQuick(indexN, indexM)/denominator)
          }
        }
      }

      val newModel:HMMModel = new HMMModel(curModel.getNumberOfHiddenStates, curModel.getNumberOfObservableSymbols, transitionMatrix, emissionMatrix, initialProbabilities)

      if (checkForConvergence(newModel, curModel, epsilon)) {
        stop = true
      }

      curModel = newModel
    }

    curModel
  }
 
  def test[K: ClassTag](model: HMMModel) = {
  }
}

object HiddenMarkovModel extends HiddenMarkovModel with java.io.Serializable

