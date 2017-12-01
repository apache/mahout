/*
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

package org.apache.mahout.drivers

import org.apache.mahout.classifier.sequencelearning.hmm.{SparkHiddenMarkovModel, _}
import org.apache.mahout.common.Hadoop2HDFSUtil
import org.apache.mahout.math.drm
import org.apache.mahout.math._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.{CheckpointedDrm, drmParallelizeEmpty}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.{`::`, dense}
import scala.collection.immutable.HashMap
import org.apache.mahout.sparkbindings._

object TrainHMMDriver extends MahoutSparkDriver {
  // define only the options specific to TrainHMM
  private final val trainHMMOptions = HashMap[String, Any](
    "numberOfHiddenStates" -> 0,
    "numberOfOfObservableSymbols" -> 0,
    "maxNumberOfIterations" -> 0,
    "appName" -> "TrainHMMDriver")

  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutSparkOptionParser(programName = "spark-trainhmm") {
      head("spark-trainhmm", "Mahout 0.10.0")

      // Input output options, non-driver specific
      note("Input, option")
      opt[String]('i', "input") required() action { (x, options) =>
        options + ("input" -> x)
      } text ("Input: path to training data " +
        " (required)")

      note("Output, option")
      opt[String]('o', "output") required() action { (x, options) =>
        options + ("output" -> x)
      } text ("Output: path to store trained model " +
        " (required)")

      // Algorithm control options--driver specific
      opts = opts ++ trainHMMOptions
      note("\nAlgorithm control options:")

      // The number of hidden states
      opt[Int]("numberOfHiddenStates") abbr "nh" required() action { (x, options) =>
        options + ("numberOfHiddenStates" -> x)
      } text "Number of hidden states" +
      trainHMMOptions("numberOfHiddenStates") validate { x =>
        if (x > 0) success else failure("Option --numberOfHiddenStates must be > 0")
      }

      opt[Int]("numberOfOfObservableSymbols") abbr "no" required()  action { (x, options) =>
        options + ("numberOfObservableSymbols" -> x)
      } text "Number of observable symbols" +
      trainHMMOptions("numberOfOfObservableSymbols") validate { x =>
        if (x > 0) success else failure("Option --numberOfOfObservableSymbols must be > 0")
      }

      // epsilon
      opts = opts + ("epsilon" -> 1.0)
      opt[Double]("epsilon") abbr "e" action { (x, options) =>
        options + ("epsilon" -> x)
      } text "Convergence threshold" validate { x =>
        if (x > 0) success else failure("Option --epsilon must be > 0")
      }

      opt[Int]("maxNumberOfIterations") abbr "n" required() action { (x, options) =>
        options + ("maxNumberOfIterations" -> x)
      } text "Maximum Number of Iterations" +
      trainHMMOptions("maxNumberOfIterations") validate { x =>
        if (x > 0) success else failure("Option --maxNumberOfIterations must be > 0")
      }
      
      // default scale is false
      opts = opts + ("scale" -> false)
      opt[Unit]("scale") abbr "s" action { (_, options) =>
        options + ("scale" -> true)
      } text "Rescale forward and backward variables after each iteration, Default: false."

      opts = opts + ("pathToInitialModel" -> "")
      opt[String]("pathToInitialModel") abbr ("pm") action { (x, options) =>
        options + ("pathToInitialModel" -> x)
      } text ("Path to the file with Initial Model parameters")

      // Spark config options--not driver specific
      parseSparkOptions()

      help("help") abbr "h" text "prints this usage text\n"

    }

    parser.parse(args, parser.opts) map { opts =>
      parser.opts = opts
      process
    }
  }

  private def readTrainingSet() : DrmLike[Long]= {
    val inputPath = parser.opts("input").asInstanceOf[String]
    var rddA = mc.textFile(inputPath).map ( line => line.split(" ") )
    .map(numbers => new DenseVector(numbers.map(_.toDouble)))

    val drmRddA: DrmRdd[Long] = rddA
                 .zipWithIndex()
                 .map(t => (t._2, t._1))
    
    val observations = drmWrap(rdd = drmRddA)
    observations
  }

  private def createInitialModel(): HMMModel = {
    val pathToModel = parser.opts("pathToInitialModel").asInstanceOf[String]
    val numberOfHiddenStates = parser.opts("numberOfHiddenStates").asInstanceOf[Int]
    val numberOfObservableSymbols = parser.opts("numberOfObservableSymbols").asInstanceOf[Int]
    var transitionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfHiddenStates)
    var emissionMatrix = new DenseMatrix(numberOfHiddenStates, numberOfObservableSymbols)
    var initialProbabilities = new DenseVector(numberOfHiddenStates)
    if (pathToModel != "")
    {
        var rddA = mc.textFile(pathToModel)
    		.map ( line => line.split(" ") )
		.map(n => new DenseVector(n.map(_.toDouble)))
		.collect

	initialProbabilities = rddA(0)
	for (index <- 1 to numberOfHiddenStates) {
	    transitionMatrix.assignRow(index - 1, rddA(index));
	}

	for (index <- numberOfHiddenStates + 1 to numberOfHiddenStates * 2) {
	    emissionMatrix.assignRow(index - numberOfHiddenStates - 1, rddA(index));
	}

	new HMMModel(numberOfHiddenStates, numberOfObservableSymbols, transitionMatrix, emissionMatrix, initialProbabilities)
    }
    else
    {
	// create random initial model
        new HMMModel(numberOfHiddenStates, numberOfObservableSymbols, transitionMatrix, emissionMatrix, initialProbabilities)
    }

  }

  override def process(): Unit = {
    start()

    val outputPath = parser.opts("output").asInstanceOf[String]
    val scale = parser.opts("scale").asInstanceOf[Boolean]
    val numberOfHiddenStates = parser.opts("numberOfHiddenStates").asInstanceOf[Int]
    val numberOfObservableSymbols = parser.opts("numberOfObservableSymbols").asInstanceOf[Int]
    val epsilon = parser.opts("epsilon").asInstanceOf[Double]
    val maxNumberOfIterations = parser.opts("maxNumberOfIterations").asInstanceOf[Int]
    
    val trainingSet = readTrainingSet()
    val initModel = createInitialModel()
    val model = SparkHiddenMarkovModel.train(initModel, trainingSet, numberOfHiddenStates, 
      numberOfObservableSymbols, epsilon, maxNumberOfIterations, scale)

    println("TM:")
    for (indexN <- 0 to model.getNumberOfHiddenStates - 1) {
      for (indexM <- 0 to model.getNumberOfHiddenStates - 1) {
        println(model.getTransitionMatrix.getQuick(indexN, indexM))
      }
    }

    println("EM:")
    for (indexN <- 0 to model.getNumberOfHiddenStates - 1) {
      for (indexM <- 0 to model.getNumberOfObservableSymbols - 1) {
        println(model.getEmissionMatrix.getQuick(indexN, indexM))
      }
    }
    model.dfsWrite(outputPath)
    
    stop()
  }

}
