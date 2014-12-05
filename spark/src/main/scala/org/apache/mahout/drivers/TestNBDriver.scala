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

import org.apache.mahout.classifier.naivebayes.{NBModel, NaiveBayes}
import org.apache.mahout.math.drm
import org.apache.mahout.math.drm. DrmLike
import scala.collection.immutable.HashMap


object TestNBDriver extends MahoutSparkDriver {
  // define only the options specific to ItemSimilarity
  private final val testNBOptipns = HashMap[String, Any](
    "appName" -> "TestNBDriver")


  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutOptionParser(programName = "spark-testnb") {
      head("spark-testnb", "Mahout 1.0")

      //Input output options, non-driver specific
      parseIOOptions(numInputs = 1)

      //Algorithm control options--driver specific
      opts = opts ++ testNBOptipns
      note("\nAlgorithm control options:")
      opt[Unit]('c',"testComplementary") hidden() action { (_, options) =>
        options + ("testComplementary" -> true) }

      opt[String]("pathToModel") abbr ("m") action { (x, options) =>
        options + ("pathToModel" -> x)
      } text ("Path to the Trained Model")


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

//    // set a large akka frame size for bcast vectors
//    sparkConf.set("spark.akka.frameSize","100")

    super.start(masterUrl, appName)

    }

  private def readTestSet: DrmLike[_] = {
    val inputPath = parser.opts("input").asInstanceOf[String]
    val trainingSet= drm.drmDfsRead(inputPath)
    trainingSet
  }
  private def readModel: NBModel = {
    val inputPath = parser.opts("pathToModel").asInstanceOf[String]
    val model= NBModel.dfsRead(inputPath)
    model
  }

  override def process: Unit = {
    start()

    val trainComplementary = parser.opts("testComplementary").asInstanceOf[Boolean]
    val outputPath = parser.opts("output").asInstanceOf[String]

    println("Reading test set...")
    val testSet = readTestSet
    println("Training model...")
    val model = readModel
    val analyzer= NaiveBayes.testNB(model, testSet)
    println(analyzer)

    stop
  }

}
//import com.esotericsoftware.kryo.Kryo
//import org.apache.spark.serializer.KryoRegistrator
//import org.apache.mahout.classifier.naivebayes.NBModel
//import org.apache.mahout.classifier.naivebayes.ComplementaryNBClassifier
//import org.apache.mahout.classifier.naivebayes.StandardNBClassifier
//class NBKryoRegistrator extends KryoRegistrator {
//  override def registerClasses(kryo: Kryo) {
//    kryo.register(classOf[NBModel])
//    kryo.register(classOf[ComplementaryNBClassifier])
//    kryo.register(classOf[StandardNBClassifier])
//  }
//}
