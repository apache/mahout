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
      scalingFactors = Some(new Array(observationSequence.length))
      var forwardVariablesTemp = new DenseMatrix(observationSequence.length, initModel.getNumberOfHiddenStates)
      // Initialization
      for (index <- 0 until initModel.getNumberOfHiddenStates) {
        forwardVariablesTemp.setQuick(0, index, initModel.getInitialProbabilities.getQuick(index)
	    			* initModel.getEmissionMatrix.getQuick(index, observationSequence(0).toInt));
      }

      var sum:Double = 0.0
      for (index <- 0 until initModel.getNumberOfHiddenStates) {
        sum += forwardVariablesTemp.getQuick(0, index)
      }

      scalingFactors.get(0) = 1.0/sum

      for (index <- 0 until initModel.getNumberOfHiddenStates) {
        forwardVariables.setQuick(0, index, forwardVariablesTemp.getQuick(0, index) * scalingFactors.get(0))
      }

      // Induction
      for (indexT <- 1 until observationSequence.length) {
        for (indexN <- 0 until initModel.getNumberOfHiddenStates) {
	  var sumA:Double = 0.0
	  for (indexM <- 0 until initModel.getNumberOfHiddenStates) {
            sumA += forwardVariables.getQuick(indexT - 1, indexM) * initModel.getTransitionMatrix.getQuick(indexM, indexN) * initModel.getEmissionMatrix.getQuick(indexN, observationSequence(indexT).toInt)
          }

          forwardVariablesTemp.setQuick(indexT, indexN, sumA)
        }

        var sumT:Double = 0.0
        for (indexN <- 0 until initModel.getNumberOfHiddenStates) {
          sumT += forwardVariablesTemp.getQuick(indexT, indexN)
        }

        scalingFactors.get(indexT) = 1.0/sumT

        for (indexN <- 0 until initModel.getNumberOfHiddenStates) {
          forwardVariables.setQuick(indexT, indexN, scalingFactors.get(indexT) * forwardVariablesTemp.getQuick(indexT, indexN))
        }
      }
    } else {
      // Initialization
      for (index <- 0 until initModel.getNumberOfHiddenStates) {
        forwardVariables.setQuick(0, index, initModel.getInitialProbabilities.getQuick(index)
	    			* initModel.getEmissionMatrix.getQuick(index, observationSequence(0).toInt));
      }

      // Induction
      for (indexT <- 1 until observationSequence.length) {
        for (indexN <- 0 until initModel.getNumberOfHiddenStates) {
	  var sum:Double = 0.0
	  for (indexM <- 0 until initModel.getNumberOfHiddenStates) {
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
      for (index <- 0 until initModel.getNumberOfHiddenStates) {
        backwardVariablesTemp.setQuick(observationSequence.length - 1, index, 1);
        backwardVariables.setQuick(observationSequence.length - 1, index, scalingFactors.get(observationSequence.length - 1) * backwardVariablesTemp.getQuick(observationSequence.length - 1, index))
      }

      // induction
      for (indexT <- observationSequence.length - 2 to 0 by -1) {
        for (indexN <- 0 until initModel.getNumberOfHiddenStates) {
	  var sum:Double = 0.0
	  for (indexM <- 0 until initModel.getNumberOfHiddenStates) {
            sum += backwardVariables.getQuick(indexT + 1, indexM) * initModel.getTransitionMatrix.getQuick(indexN, indexM) * initModel.getEmissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt) 
          }

          backwardVariablesTemp.setQuick(indexT, indexN, sum)
          backwardVariables.setQuick(indexT, indexN, backwardVariablesTemp.getQuick(indexT, indexN) * scalingFactors.get(indexT))
        }
      }
    } else {
      // Initialization
      for (index <- 0 until initModel.getNumberOfHiddenStates) {
        backwardVariables.setQuick(observationSequence.length - 1, index, 1)
      }
      // Induction
      for (indexT <- observationSequence.length - 2 to 0 by -1) {
        for (indexN <- 0 until initModel.getNumberOfHiddenStates) {
	  var sum:Double = 0.0
	  for (indexM <- 0 until initModel.getNumberOfHiddenStates) {
	  	      sum += backwardVariables.getQuick(indexT + 1, indexM) * initModel.getTransitionMatrix.getQuick(indexN, indexM) * initModel.getEmissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt)
	  }

          backwardVariables.setQuick(indexT, indexN, sum)
	}
      }
    }

    backwardVariables
  }

  def expectedNumberOfTransitions(model: HMMModel, observationSequence:Vector, forwardVariables: DenseMatrix,
    backwardVariables: DenseMatrix, likelihood: Double, indexN: Int, indexM: Int, scale: Boolean): Double = {
    var numTransitions:Double = 0.0
    for (indexT <- 0 until observationSequence.length - 1) {
      numTransitions += forwardVariables.getQuick(indexT, indexN) * model.getEmissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt) * backwardVariables.getQuick(indexT + 1, indexM)
    }

    if (scale) {
      numTransitions = (numTransitions * model.getTransitionMatrix.getQuick(indexN, indexM))
    } else {
      numTransitions = (numTransitions * model.getTransitionMatrix.getQuick(indexN, indexM)) / likelihood
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
  ): HMMModel = {
    var curModel:HMMModel = initModel
    var iter = 0
    var stop = false
    val observationsMatrix = observations.collect

    while ((iter < maxNumberOfIterations) && (!stop)) {
      iter = iter + 1
      var transitionMatrix = new DenseMatrix(curModel.getNumberOfHiddenStates, curModel.getNumberOfHiddenStates)
      var emissionMatrix = new DenseMatrix(curModel.getNumberOfHiddenStates, curModel.getNumberOfObservableSymbols)
      var initialProbabilities = new DenseVector(curModel.getNumberOfHiddenStates)

      for (indexN <- 0 until curModel.getNumberOfHiddenStates) {
        initialProbabilities.setQuick(indexN, 0.0)
        for (indexM <- 0 until curModel.getNumberOfHiddenStates) {
          transitionMatrix.setQuick(indexN, indexM, 0.0)
        }
        
        for (indexM <- 0 until curModel.getNumberOfObservableSymbols) {
          emissionMatrix.setQuick(indexN, indexM, 0.0)          
        }
      }

      for (obsIndex <- 0 until observationsMatrix.nrow) {
        val observation = observationsMatrix.viewRow(obsIndex)
        val (forwardVariables, scalingFactors) = computeForwardVariables(curModel, observation, scale)
        val backwardVariables = computeBackwardVariables(curModel, observation, scale, scalingFactors)
        val obsLikelihood = sequenceLikelihood(forwardVariables, scalingFactors)
      
        // recompute initial probabilities
        if (scale) {
          for (index <- 0 until curModel.getNumberOfHiddenStates) {
            val prob:Double = forwardVariables.getQuick(0, index) * backwardVariables.getQuick(0, index) / scalingFactors.get(0)
            val oldVal:Double = initialProbabilities.getQuick(index)
            initialProbabilities.setQuick(index, prob + oldVal)
          }
        } else {
          for (index <- 0 until curModel.getNumberOfHiddenStates) {
            val prob:Double = forwardVariables.getQuick(0, index) * backwardVariables.getQuick(0, index) / obsLikelihood
            val oldVal:Double = initialProbabilities.getQuick(index)
            initialProbabilities.setQuick(index, prob + oldVal)
          }
        }

        // recompute transitionmatrix
        for (indexN <- 0 until curModel.getNumberOfHiddenStates) {
          for (indexM <- 0 until curModel.getNumberOfHiddenStates) {
            val numTransitions = expectedNumberOfTransitions(curModel, observation, forwardVariables, backwardVariables, obsLikelihood, indexN, indexM, scale)
            val oldVal:Double = transitionMatrix.getQuick(indexN, indexM)
            transitionMatrix.setQuick(indexN, indexM, numTransitions + oldVal)
          }
        }

        // recompute emissionmatrix
        for (indexN <- 0 until curModel.getNumberOfHiddenStates) {
          var denominator:Double = 0.0
          for (indexM <- 0 until curModel.getNumberOfObservableSymbols) {
            var numEmissions:Double = expectedNumberOfEmissions(observation, forwardVariables, backwardVariables, obsLikelihood, indexN, indexM, scale, scalingFactors)
            val oldVal:Double = emissionMatrix.getQuick(indexN, indexM)
            emissionMatrix.setQuick(indexN, indexM, numEmissions + oldVal)
          }
        }
      }

      // normalize the matrices
      for (indexN <- 0 until curModel.getNumberOfHiddenStates) {
        val normI:Double = initialProbabilities.getQuick(indexN)/observationsMatrix.nrow
        initialProbabilities.setQuick(indexN, normI)

        var total:Double = 0.0
        for (indexM <- 0 until curModel.getNumberOfHiddenStates) {
          total += transitionMatrix.getQuick(indexN, indexM)
        }

        for (indexM <- 0 until curModel.getNumberOfHiddenStates) {
          transitionMatrix.setQuick(indexN, indexM, transitionMatrix.getQuick(indexN, indexM)/total)
        }

        total = 0.0
        for (indexM <- 0 until curModel.getNumberOfObservableSymbols) {
          total += emissionMatrix.getQuick(indexN, indexM)
        }

        for (indexM <- 0 until curModel.getNumberOfObservableSymbols) {
          emissionMatrix.setQuick(indexN, indexM, emissionMatrix.getQuick(indexN, indexM)/total)
        }
      }

      val newModel:HMMModel = new HMMModel(curModel.getNumberOfHiddenStates, curModel.getNumberOfObservableSymbols, transitionMatrix, emissionMatrix, initialProbabilities)

      if (checkForConvergence(curModel, newModel, epsilon)) {
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

