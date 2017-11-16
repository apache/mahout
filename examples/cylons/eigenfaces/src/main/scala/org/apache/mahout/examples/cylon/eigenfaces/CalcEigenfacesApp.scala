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

package org.apache.mahout.examples.cylon.eigenfaces

import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import org.apache.mahout.math._
import org.apache.mahout.math.algorithms.preprocessing.MeanCenter
import org.apache.mahout.math.decompositions._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.sparkbindings._
import org.apache.spark.{SparkConf, SparkContext}
import org.rawkintrevo.cylon.common.mahout.MahoutUtils
import org.rawkintrevo.cylon.frameprocessors.OpenCVImageUtils

object CalcEigenfacesApp {
  def main(args: Array[String]): Unit = {
    val cylon_home = scala.util.Properties.envOrElse("CYLON_HOME", "../" )
    case class Config(
                       inputDirectory: String = cylon_home + "/data/lfw-deepfunneled",
                       outputDirectory: String = cylon_home + "/data/eigenfaces",
                       droneName: String = "test",
                       parallelism: Int = 50,
                       kEigenfaces: Int = 130
                     )

    val parser = new scopt.OptionParser[Config]("scopt") {
      head("Eigenface Calculator", "1.0-SNAPSHOT")

      opt[String]('i', "inputDirectory").optional()
        .action((x, c) => c.copy(inputDirectory = x))
        .text("Input Directory of lfw-deepfunneled. Default: $CYLON_HOME/data/lfw-deepfunneled")

      opt[String]('o', "outputDirectory").optional()
        .action((x, c) => c.copy(outputDirectory = x))
        .text("Output directory to save results to. Default: $CYLON_HOME/data/eigenfaces ")

      opt[Int]('p', "parallelism. default: 50").optional()
        .action((x, c) => c.copy(parallelism = x))

      opt[Int]('k', "number of eigenfaces to produces. default: 130").optional()
        .action((x, c) => c.copy(kEigenfaces = x))

      help("help").text("prints this usage text")
    }

    parser.parse(args, Config()) map { config =>

      val sparkConf = new SparkConf()
        .setAppName("Calculate Eigenfaces")
        .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.kryo.referenceTracking", "false")
        .set("spark.kryoserializer.buffer", "32k")
        .set("spark.kryoserializer.buffer.max", "1g")
        .set("spark.executor.memory", "16g")
        .set("spark.driver.memory", "2g")


      val sc = new SparkContext(sparkConf)

      implicit val sdc: org.apache.mahout.sparkbindings.SparkDistributedContext = sc2sdc(sc)

      val par = config.parallelism // When using OMP you want as little parallelization as possible
      // todo utilize CYLON_HOME

      val imagesRDD: DrmRdd[Int] = sc.binaryFiles(config.inputDirectory + "/*/*", par)
        .map(o => new DenseVector(
          OpenCVImageUtils.bufferedImageToDoubleArray(
            ImageIO.read(new ByteArrayInputStream(o._2.toArray())))))
        .zipWithIndex
        .map(o => (o._2.toInt, o._1)) // Make the Index the first value of the tuple


      val imagesDRM = drmWrap(rdd = imagesRDD).checkpoint()

      println(s"Dataset: ${imagesDRM.nrow} images, ${imagesDRM.ncol} pixels per image")

      // Mean Center Pixels
      val mcModel = new MeanCenter().fit(imagesDRM)
      // mcModel.colCentersV need to persist this out to be loaded by streaming model.

      val colMeansInCore = dense(mcModel.colCentersV)

      val mcImagesDrm = mcModel.transform(imagesDRM)

      val numberOfEigenfaces = config.kEigenfaces
      val (drmU, drmV, s) = dssvd(mcImagesDrm, k = numberOfEigenfaces, p = 15, q = 0)

      /**
        * drmV -> Eignfaces (transposed) need to load this into Flink engine
        * drmU -> Eigenface linear combos of input faces, load this into Solr
        * -- Or don't only required if we're going to match celebrities.
        */


      // Todo: make this a config (at least the directory)

        MahoutUtils.matrixWriter(colMeansInCore, config.outputDirectory + "/colMeans.mmat")

        import org.apache.mahout.math.scalabindings.MahoutCollections._
        //drmV.rdd.map(row => row._2.toArray.mkString(",")).coalesce(1).saveAsTextFile(config.outputDirectory + "/eigenfaces.mmat")
        MahoutUtils.matrixWriter(drmV.collect, config.outputDirectory + "/eigenfaces.mmat")

      // Stupid- can't read it into flink if you do this
//      drmParallelize(colMeansInCore, 1).dfsWrite("file:///home/rawkintrevo/gits/cylon-blog/data/colMeans")
//      drmParallelize(drmV, 3).dfsWrite("file:///home/rawkintrevo/gits/cylon-blog/data/eigenfaces")
    } getOrElse {
      // arguments are bad, usage message will have been displayed
    }
  }
}
