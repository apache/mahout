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

import org.apache.mahout.math._

import org.apache.mahout.math.{drm, scalabindings}

import scalabindings._
import scalabindings.RLikeOps._
import drm._
import scala.language.asInstanceOf
import scala.collection._
import JavaConversions._
import scala.util.Random
import org.apache.mahout.common.RandomUtils

/**
 *
 * @param numHiddenStates number of hidden states
 * @param numOutputStates number of output states
 */
class HMMModel(val numberOfHiddenStates: Int,
               val numberOfOutputSymbols: Int,
	       var transitionMatrix: Matrix = null,
	       var emissionMatrix: Matrix = null,
  	       var initialProbabilities: Vector = null)  extends java.io.Serializable {

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
          sum = sum + transitionMatrix.getQuick(i, j)
          cumulativeTransitionMatrix.setQuick(i, j, sum)
        }
        cumulativeTransitionMatrix.setQuick(i, numberOfHiddenStates - 1, 1.0)
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
          sum = sum + emissionMatrix.getQuick(i, j)
          cumulativeEmissionMatrix.setQuick(i, j, sum)
        }
        cumulativeEmissionMatrix.setQuick(i, numberOfOutputSymbols - 1, 1.0)
      }
    }

    cumulativeEmissionMatrix
  }

  def getCumulativeInitialProbabilities: Vector = {
    if (cumulativesInitialized == false) {
      cumulativeInitialProbabilities = new DenseVector(numberOfHiddenStates)
      var sum:Double = 0;
      for (i <- 0 until numberOfHiddenStates) {
        sum = sum + initialProbabilities.getQuick(i)
        cumulativeInitialProbabilities.setQuick(i, sum)
      }
      cumulativeInitialProbabilities.setQuick(numberOfHiddenStates - 1, 1.0)
    }

    cumulativeInitialProbabilities
  }

  def printModel(): Unit = {
    println("Transition Matrix:")
    println(getTransitionMatrix)
    println("Emission Matrix:")
    println(getEmissionMatrix)
    println("Initial Probabilities Vector:")
    println(getInitialProbabilities)
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
      initialProbabilities.setQuick(i, nextRand);
      sum += nextRand;
    }
    // "normalize" the vector to generate probabilities
    for (i <- 0 to numberOfHiddenStates - 1) {
      initialProbabilities.setQuick(i, initialProbabilities.getQuick(i)/sum);
    }

    // initialize the transition matrix
    for (i <- 0 to numberOfHiddenStates -1) {
      sum = 0
      for (j <- 0 to numberOfHiddenStates - 1) {
        transitionMatrix.setQuick(i, j, rand.nextDouble())
        sum += transitionMatrix.getQuick(i, j)
      }
      // normalize the random values to obtain probabilities
      for (j <- 0 to numberOfHiddenStates - 1) {
        transitionMatrix.setQuick(i, j, transitionMatrix.getQuick(i, j)/sum)
      }
    }

    // initialize the output matrix
    for (i <- 0 to numberOfHiddenStates - 1) {
      sum = 0
      for (j <- 0 to numberOfOutputSymbols - 1) {
        emissionMatrix.setQuick(i, j, rand.nextDouble())
        sum += emissionMatrix.getQuick(i, j)
      }
      // normalize the random values to obtain probabilities
      for (j <- 0 to numberOfOutputSymbols - 1) {
        emissionMatrix.setQuick(i, j, emissionMatrix.getQuick(i, j)/sum)
      }
    }
  }

  /**
   * Write a trained model to the filesystem as a series of DRMs
   * @param pathToModel Directory to which the model will be written
   */
  def dfsWrite(pathToModel: String)(implicit ctx: DistributedContext): Unit = {
    val fullPathToModel = pathToModel + HMMModel.modelBaseDirectory
    drmParallelize(sparse(svec((0, numberOfHiddenStates)::Nil))).dfsWrite(fullPathToModel + "/numberOfHiddenStatesDrm.drm")
    drmParallelize(sparse(svec((0, numberOfOutputSymbols)::Nil))).dfsWrite(fullPathToModel + "/numberOfOutputSymbolsDrm.drm")
    drmParallelize(transitionMatrix).dfsWrite(fullPathToModel + "/transitionMatrixDrm.drm")
    drmParallelize(emissionMatrix).dfsWrite(fullPathToModel + "/emissionMatrixDrm.drm")
    drmParallelize(sparse(initialProbabilities)).dfsWrite(fullPathToModel + "/initialProbabilitiesDrm.drm")
  }

  /** Model Validation */
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
      assert(initialProbabilities.getQuick(i) >= 0, "Initial probability of state is negative")
      sum += initialProbabilities.getQuick(i)
    }

    assert(Math.abs(sum - 1) <= 0.000001, "Initial probabilities do not add up to 1")

    for (i <- 0 until numberOfHiddenStates) {
      sum = 0
      for (j <- 0 until numberOfHiddenStates) {
        assert(transitionMatrix.getQuick(i, j) >= 0, "transition probability is negative")
        sum += transitionMatrix.getQuick(i, j)
      }
      assert(Math.abs(sum - 1) <= 0.000001, "transition probabilities do not add up to 1")
    }

    for (i <- 0 until numberOfHiddenStates) {
      sum = 0
      for (j <- 0 until numberOfOutputSymbols) {
        assert(emissionMatrix.getQuick(i, j) >= 0, "emission probability is negative")
        sum += emissionMatrix.getQuick(i, j)
      }
      assert(Math.abs(sum - 1) <= 0.000001, "emission probabilities do not add up to 1")
    }
  }
}

object HMMModel extends java.io.Serializable {

  val modelBaseDirectory = "/hiddenMarkovModel"
  
  def dfsRead(pathToModel: String)(implicit ctx: DistributedContext): HMMModel = {
    // read from a base directory for all drms
    val fullPathToModel = pathToModel + HMMModel.modelBaseDirectory
    val numberOfHiddenStatesDrm = drmDfsRead(fullPathToModel + "/numberOfHiddenStatesDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val numberOfHiddenStates = numberOfHiddenStatesDrm.collect(0, 0).toInt
    numberOfHiddenStatesDrm.uncache()
    val numberOfOutputSymbolsDrm = drmDfsRead(fullPathToModel + "/numberOfOutputSymbolsDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val numberOfOutputSymbols = numberOfOutputSymbolsDrm.collect(0, 0).toInt
    numberOfOutputSymbolsDrm.uncache()
    val transitionMatrixDrm = drmDfsRead(fullPathToModel + "/transitionMatrixDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val transitionMatrix = transitionMatrixDrm.collect
    transitionMatrixDrm.uncache()
    val emissionMatrixDrm = drmDfsRead(fullPathToModel + "/emissionMatrixDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val emissionMatrix = emissionMatrixDrm.collect
    emissionMatrixDrm.uncache()
    val initialProbabilitiesDrm = drmDfsRead(fullPathToModel + "/initialProbabilitiesDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val initialProbabilities = initialProbabilitiesDrm.collect(0, ::)
    initialProbabilitiesDrm.uncache()
    val model: HMMModel = new HMMModel(numberOfHiddenStates, numberOfOutputSymbols, transitionMatrix, emissionMatrix, initialProbabilities)

    model
  }
}
