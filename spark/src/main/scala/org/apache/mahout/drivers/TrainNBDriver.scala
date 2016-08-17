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

import org.apache.mahout.classifier.naivebayes.{SparkNaiveBayes, _}
import org.apache.mahout.common.Hadoop1HDFSUtil
import org.apache.mahout.math.drm
import org.apache.mahout.math.drm.DrmLike

import scala.collection.immutable.HashMap


object TrainNBDriver extends MahoutSparkDriver {
  // define only the options specific to TrainNB
  private final val trainNBOptipns = HashMap[String, Any](
    "appName" -> "TrainNBDriver")

  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutSparkOptionParser(programName = "spark-trainnb") {
      head("spark-trainnb", "Mahout 0.10.0")

      // Input output options, non-driver specific
      parseIOOptions(numInputs = 1)

      // Algorithm control options--driver specific
      opts = opts ++ trainNBOptipns
      note("\nAlgorithm control options:")

      // default trainComplementary is false
      opts = opts + ("trainComplementary" -> false)
      opt[Unit]("trainComplementary") abbr "c" action { (_, options) =>
        options + ("trainComplementary" -> true)
      } text "Train a complementary model, Default: false."

      // Laplace smoothing paramater default is 1.0
      opts = opts + ("alphaI" -> 1.0)
      opt[Double]("alphaI") abbr "a" action { (x, options) =>
        options + ("alphaI" -> x)
      } text "Laplace smothing factor default is 1.0" validate { x =>
        if (x > 0) success else failure("Option --alphaI must be > 0")
      }

      // Overwrite the output directory (with the model) if it exists?  Default: false
      opts = opts + ("overwrite" -> false)
      opt[Unit]("overwrite") abbr "ow" action { (_, options) =>
        options + ("overwrite" -> true)
      } text "Overwrite the output directory (with the model) if it exists? Default: false"

      // Spark config options--not driver specific
      parseSparkOptions()

      help("help") abbr "h" text "prints this usage text\n"

    }
    parser.parse(args, parser.opts) map { opts =>
      parser.opts = opts
      process
    }
  }

  /** Read the training set from inputPath/part-x-00000 sequence file of form <Text,VectorWritable> */
  private def readTrainingSet(): DrmLike[_]= {
    val inputPath = parser.opts("input").asInstanceOf[String]
    val trainingSet= drm.drmDfsRead(inputPath)
    trainingSet
  }

  override def process(): Unit = {
    start()

    val complementary = parser.opts("trainComplementary").asInstanceOf[Boolean]
    val outputPath = parser.opts("output").asInstanceOf[String]
    val alpha = parser.opts("alphaI").asInstanceOf[Double]
    val overwrite = parser.opts("overwrite").asInstanceOf[Boolean]

    val fullPathToModel = outputPath + NBModel.modelBaseDirectory

    if (overwrite) {
       Hadoop1HDFSUtil.delete(fullPathToModel)
    }

    val trainingSet = readTrainingSet()
    // Use Spark-Optimized Naive Bayes here to extract labels and aggregate options
    val (labelIndex, aggregatedObservations) = SparkNaiveBayes.extractLabelsAndAggregateObservations(trainingSet)
    val model = SparkNaiveBayes.train(aggregatedObservations, labelIndex, complementary, alpha.toFloat)

    model.dfsWrite(outputPath)

    stop()
  }

}
