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

import org.apache.mahout.classifier.naivebayes._
import org.apache.mahout.classifier.naivebayes.SparkNaiveBayes
import org.apache.mahout.common.HDFSPathSearch
import org.apache.mahout.math.cf.SimilarityAnalysis
import org.apache.mahout.math.drm
import org.apache.mahout.math.drm.{DistributedContext, DrmLike}
import org.apache.mahout.math.indexeddataset.{Schema, IndexedDataset, indexedDatasetDFSReadElements}
import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark
import scala.collection.immutable.HashMap


object TrainAndTestNBDriver extends MahoutSparkDriver {
  // define only the options specific to ItemSimilarity
  private final val trainNBOptipns = HashMap[String, Any](
    "appName" -> "TrainNBDriver")

  private var writeSchema: Schema = _
  private var readSchema1: Schema = _
  private var readSchema2: Schema = _

  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutOptionParser(programName = "spark-trainnb") {
      head("spark-trainnb", "Mahout 1.0")

      //Input output options, non-driver specific
      parseIOOptions(numInputs = 1)

      //Algorithm control options--driver specific
      opts = opts ++ trainNBOptipns
      note("\nAlgorithm control options:")
      opt[Unit]('c',"trainComplementary") hidden() action { (_, options) =>
        options + ("trainComplementary" -> true) }


      //How to search for input
      parseFileDiscoveryOptions

      //Drm output schema--not driver specific, drm specific
      parseDrmFormatOptions

      //Spark config options--not driver specific
      parseSparkOptions

      //Jar inclusion, this option can be set when executing the driver from compiled code, not when from CLI
      parseGenericOptions

      help("help") abbr ("h") text ("prints this usage text\n")

    }
    parser.parse(args, parser.opts) map { opts =>
      parser.opts = opts
      process
    }
  }

  override def start(masterUrl: String = parser.opts("master").asInstanceOf[String],
      appName: String = parser.opts("appName").asInstanceOf[String]):
    Unit = {

    // will be only specific to this job.
    sparkConf.set("spark.kryo.referenceTracking", "false")
      .set("spark.kryoserializer.buffer.mb", "50")// todo: should this be left to config or an option?

    if (parser.opts("sparkExecutorMem").asInstanceOf[String] != "")
      sparkConf.set("spark.executor.memory", parser.opts("sparkExecutorMem").asInstanceOf[String])

    // set a large akka frame size
    //sparkConf.set("spark.akka.frameSize","20") // don't need this for Spark optimized NaiveBayes..
    //else leave as set in Spark config

    super.start(masterUrl, appName)

    }

  private def readTrainingSet: DrmLike[_]= {
    val inputPath = parser.opts("input").asInstanceOf[String]
    val trainingSet= drm.drmDfsRead(inputPath)
    trainingSet
  }

  override def process: Unit = {
    start()

    val trainComplementary = parser.opts("trainComplementary").asInstanceOf[Boolean]
    val outputPath = parser.opts("output").asInstanceOf[String]

    printf("Reading training set...")
    val trainingSet = readTrainingSet
    printf("Aggregating training set and extracting labels...")
    val (labelIndex, aggregatedObservations) = SparkNaiveBayes.extractLabelsAndAggregateObservations(trainingSet)
    printf("Training model...")
    val model = NaiveBayes.trainNB(aggregatedObservations, labelIndex)
    printf("Saving model to "+outputPath+"...")
    model.dfsWrite(outputPath)



    stop
  }

}
