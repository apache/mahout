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
  ): (Matrix, Option[Array[Double]]) = {
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
  ): Matrix = {
    var backwardVariables = new DenseMatrix(0,0)
    if (scale)
    {
      var backwardVariablesTemp = new DenseMatrix(observationSequence.length, initModel.getNumberOfHiddenStates)
      // initialization
      for (index <- 0 to initModel.getNumberOfHiddenStates - 1) {
        backwardVariablesTemp.setQuick(observationSequence.length - 1, index, 1);
        backwardVariables.setQuick(observationSequence.length - 1, index, scalingFactors.get(observationSequence.length - 1) * backwardVariablesTemp.getQuick(observationSequence.length - 1, index))
      }

      // induction
      for (indexT <- observationSequence.length - 2 to 0) {
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
      for (indexT <- observationSequence.length - 2 to 0) {
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
  
  def train(initModel: HMMModel,
    observations: DrmLike[Long],
    numberOfHiddenStates:Int,
    numberOfObservableSymbols:Int,
    epsilon: Double,
    maxNumberOfIterations:Int,
    scale: Boolean = false
  ): HMMModel = {

    var curModel = initModel
    for (index <- 0 to maxNumberOfIterations - 1) {
      var transitionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfHiddenStates)
      var emissionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfObservableSymbols)
      var initialProbabilities = new DenseVector(numberOfHiddenStates)

      curModel = new HMMModel(numberOfHiddenStates, numberOfObservableSymbols, transitionMatrix, emissionMatrix, initialProbabilities)
      val observationsMatrix = observations.collect
      val observation = observationsMatrix.viewRow(0)
      val (forwardVariables, scalingFactors) = computeForwardVariables(initModel, observation, scale)
      val backwardVariables = computeBackwardVariables(initModel, observation, scale, scalingFactors)
    }

    curModel
  }
 
  def test[K: ClassTag](model: HMMModel) = {
  }
}

object HiddenMarkovModel extends HiddenMarkovModel with java.io.Serializable

