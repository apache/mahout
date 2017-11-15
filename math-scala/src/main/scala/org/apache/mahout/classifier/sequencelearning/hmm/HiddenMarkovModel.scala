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
	    }

	    curModel
  }

  

 
  def test[K: ClassTag](model: HMMModel
                        ) = {

    
  }

  

}

object HiddenMarkovModel extends HiddenMarkovModel with java.io.Serializable
