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
    var forwardVariables = new DenseMatrix(initModel.getNumberOfHiddenStates, observationSequence.length)
    var scalingFactors = None: Option[Array[Double]]

    if (scale) {
      scalingFactors = Some(new Array(observationSequence.length));
      var forwardVariablesTemp = new DenseMatrix(initModel.getNumberOfHiddenStates, observationSequence.length)
      // Initialization
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        forwardVariablesTemp.setQuick(index, 0, initModel.getInitialProbabilities.getQuick(index)
	    			* initModel.getEmissionMatrix.getQuick(index, observationSequence(0).toInt));
      }

      var sum:Double = 0.0
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        sum += forwardVariablesTemp.getQuick(index, 0)
      }

      scalingFactors.get(0) = 1.0/sum

      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        forwardVariables.setQuick(index, 0, forwardVariablesTemp.getQuick(index, 0) * scalingFactors.get(0))
      }

      // Induction
      for (indexT <- 1 to observationSequence.length - 1) {
        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
	  var sumA:Double = 0.0
	  for (indexM <- 0 to initModel.getNumberOfHiddenStates - 1) {
            sumA += forwardVariables.getQuick(indexM, indexT - 1) * initModel.getTransitionMatrix.getQuick(indexM, indexN) * initModel.getEmissionMatrix.getQuick(indexN, observationSequence(indexT).toInt)
          }

          forwardVariablesTemp.setQuick(indexN, indexT, sumA)
        }

        var sumT:Double = 0.0
        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
          sumT += forwardVariablesTemp.getQuick(indexN, indexT)
        }

        scalingFactors.get(indexT) = 1.0/sumT

        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
          forwardVariables.setQuick(indexN, indexT, scalingFactors.get(indexT) * forwardVariablesTemp.getQuick(indexN, indexT))
        }
      }
    } else {
      // Initialization
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        forwardVariables.setQuick(index, 0, initModel.getInitialProbabilities.getQuick(index)
	    			* initModel.getEmissionMatrix.getQuick(index, observationSequence(0).toInt));
      }

      // Induction
      for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
        for (indexT <- 1 to observationSequence.length - 1) {
	  var sum:Double = 0.0
	  for (indexM <- 0 to initModel.getNumberOfHiddenStates - 1) {
	    sum += forwardVariables.getQuick(indexM, indexT - 1) * initModel.getTransitionMatrix.getQuick(indexM, indexN);
	  }

	  forwardVariables.setQuick(indexN, indexT, sum * initModel.getEmissionMatrix.getQuick(indexN, observationSequence(indexT).toInt))
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
    var backwardVariables = new DenseMatrix(initModel.getNumberOfHiddenStates, observationSequence.length)
    if (scale)
    {
      var backwardVariablesTemp = new DenseMatrix(initModel.getNumberOfHiddenStates, observationSequence.length)
      // initialization
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        backwardVariablesTemp.setQuick(index, observationSequence.length - 1, 1);
        backwardVariables.setQuick(index, observationSequence.length - 1, scalingFactors.get(observationSequence.length - 1) * backwardVariablesTemp.getQuick(index, observationSequence.length - 1))
      }

      // induction
      for (indexT <- observationSequence.length - 2 to 0) {
        for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
	  var sum:Double = 0.0
	  for (indexM <- 0 to initModel.getNumberOfHiddenStates - 1) {
            sum += backwardVariables.getQuick(indexM, indexT + 1) * initModel.getTransitionMatrix.getQuick(indexN, indexM) * initModel.getEmissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt) 
          }

          backwardVariablesTemp.setQuick(indexN, indexT, sum)
          backwardVariables.setQuick(indexN, indexT, backwardVariablesTemp.getQuick(indexN, indexT) * scalingFactors.get(indexT))
        }
      }
    } else {
      // Initialization
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        backwardVariables.setQuick(index, observationSequence.length - 1, 1)
      }
      // Induction
      for (indexN <- 0 to initModel.getNumberOfHiddenStates - 1) {
        for (indexT <- observationSequence.length - 2 to 0) {
	  var sum:Double = 0.0
	  for (indexM <- 0 to initModel.getNumberOfHiddenStates - 1) {
	  	      sum += backwardVariables.getQuick(indexM, indexT + 1) * initModel.getTransitionMatrix.getQuick(indexN, indexM) * initModel.getEmissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt)
	  }

          backwardVariables.setQuick(indexN, indexT, sum)
	}
      }
    }

    backwardVariables
  }

  def computeXiVariable(model: HMMModel, observationSequence:Vector, forwardVariables: DenseMatrix,
    backwardVariables: DenseMatrix, likelihood: Double, indexN: Int, indexM: Int, indexT:Int): Double = {
    forwardVariables.getQuick(indexN, indexT) * model.getTransitionMatrix.getQuick(indexN, indexM) * model.getEmissionMatrix.getQuick(indexM, observationSequence(indexT + 1).toInt) * backwardVariables.getQuick(indexM, indexT + 1)/likelihood
    0
  }

  def computeGammaVariable(forwardVariables: DenseMatrix,
    backwardVariables: DenseMatrix, likelihood: Double, indexN: Int, indexT: Int): Double = {
    forwardVariables.getQuick(indexN, indexT) * backwardVariables.getQuick(indexN, indexT)/likelihood
  }

  def sequenceLikelihood(forwardVariables: DenseMatrix,
    scalingFactors: Option[Array[Double]]
  ): Double = {
    var likelihood: Double = 0.0

    if (scalingFactors == None) {
      for (indexN <- 0 to forwardVariables.rowSize() - 1) {
        likelihood += forwardVariables.getQuick(indexN, forwardVariables.columnSize() - 1)
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

  def train(initModel: HMMModel,
    observations: DrmLike[Long],
    numberOfHiddenStates:Int,
    numberOfObservableSymbols:Int,
    epsilon: Double,
    maxNumberOfIterations:Int,
    scale: Boolean = false
  ): HMMModel = {
    var curModel:HMMModel = initModel

    for (index <- 0 to maxNumberOfIterations - 1) {
      var transitionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfHiddenStates)
      var emissionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfObservableSymbols)
      var initialProbabilities = new DenseVector(numberOfHiddenStates)

      val observationsMatrix = observations.collect
      val observation = observationsMatrix.viewRow(0)
      val (forwardVariables, scalingFactors) = computeForwardVariables(curModel, observation, scale)
      val backwardVariables = computeBackwardVariables(curModel, observation, scale, scalingFactors)

      if (scale) {

      } else {
        val obsLikelihood = sequenceLikelihood(forwardVariables, None)
        // recompute initial probabilities
        for (index <- 0 to curModel.getNumberOfHiddenStates - 1) {
          initialProbabilities.setQuick(index, forwardVariables.getQuick(index, 0) * backwardVariables.getQuick(index, 0) / obsLikelihood)
        }

        // recompute transitionmatrix
        for (indexN <- 0 to curModel.getNumberOfHiddenStates - 1) {
          for (indexM <- 0 to curModel.getNumberOfHiddenStates - 1) {
            var numerator:Double = 0.0
            var denominator:Double = 0.0            
            for (indexT <- 0 to observation.length - 2) {
              numerator += computeXiVariable(curModel, observation, forwardVariables, backwardVariables, obsLikelihood, indexN, indexM, indexT)

              denominator += computeGammaVariable(forwardVariables, backwardVariables, obsLikelihood, indexN, indexT)
            }

            transitionMatrix.setQuick(indexN, indexM, numerator/denominator)
          }
        }
        println("===")
        println(transitionMatrix)

        // recompute emissionmatrix
        for (indexN <- 0 to curModel.getNumberOfHiddenStates - 1) {
          for (indexM <- 0 to curModel.getNumberOfObservableSymbols - 1) {
            var numerator:Double = 0.0
            var denominator:Double = 0.0

            for (indexT <- 0 to observation.length - 1) {
              if (observation(indexT).toInt == indexM) {
                numerator += forwardVariables.getQuick(indexN, indexT) * backwardVariables.getQuick(indexN, indexT)
              }

              denominator += forwardVariables.getQuick(indexN, indexT) * backwardVariables.getQuick(indexN, indexT)
            }
            println(numerator)
            println(denominator)
            println(indexN)
            println(indexM)            
            emissionMatrix.setQuick(indexN, indexM, numerator/denominator)
          }
        }
        println("===")
        println(emissionMatrix)
        curModel = new HMMModel(numberOfHiddenStates, numberOfObservableSymbols, transitionMatrix, emissionMatrix, initialProbabilities)
      }
    }

    curModel
  }
 
  def test[K: ClassTag](model: HMMModel) = {
  }
}

object HiddenMarkovModel extends HiddenMarkovModel with java.io.Serializable

