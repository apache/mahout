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

import org.apache.mahout.classifier.naivebayes.{SparkNaiveBayes, NBModel}
import org.apache.mahout.math.drm
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.RLikeDrmOps.drm2RLikeOps
import scala.collection.immutable.HashMap


object TestNBDriver extends MahoutSparkDriver {
  // define only the options specific to TestNB
  private final val testNBOptipns = HashMap[String, Any](
    "appName" -> "TestNBDriver")

  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutSparkOptionParser(programName = "spark-testnb") {
      head("spark-testnb", "Mahout 0.10.0")

      // Input options, non-driver specific
      // we have no output except the confusion matrix to stdout so we don't need an
      // output option

      note("Input, option")
      opt[String]('i', "input") required() action { (x, options) =>
        options + ("input" -> x)
      } text ("Input: path to test data " +
        " (required)")

      // Algorithm control options--driver specific
      opts = opts ++ testNBOptipns
      note("\nAlgorithm control options:")

      // default testComplementary is false
      opts = opts + ("testComplementary" -> false)
      opt[Unit]("testComplementary") abbr ("c") action { (_, options) =>
        options + ("testComplementary" -> true)
      } text ("Test a complementary model, Default: false.")


      opt[String]("pathToModel") abbr ("m") action { (x, options) =>
        options + ("pathToModel" -> x)
      } text ("Path to the Trained Model")

      // Spark config options--not driver specific
      parseSparkOptions()

      help("help") abbr ("h") text ("prints this usage text\n")

    }
    parser.parse(args, parser.opts) map { opts =>
      parser.opts = opts
      process()
    }
  }

  /** Read the test set from inputPath/part-x-00000 sequence file of form <Text,VectorWritable> */
  private def readTestSet: DrmLike[_] = {
    val inputPath = parser.opts("input").asInstanceOf[String]
    val trainingSet = drm.drmDfsRead(inputPath)
    trainingSet
  }

  /** read the model from pathToModel using NBModel.DfsRead(...) */
  private def readModel: NBModel = {
    val inputPath = parser.opts("pathToModel").asInstanceOf[String]
    val model = NBModel.dfsRead(inputPath)
    model
  }

  override def process(): Unit = {
    start()

    val testComplementary = parser.opts("testComplementary").asInstanceOf[Boolean]

    // todo:  get the -ow option in to check for a model in the path and overwrite if flagged.

    val testSet = readTestSet
    val model = readModel
    val analyzer = SparkNaiveBayes.test(model, testSet, testComplementary)

    println(analyzer)

    stop()
  }

}

