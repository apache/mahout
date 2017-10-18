package org.rawkintrevo.cylon.flinkengine.apps

/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.awt.image.BufferedImage
import java.util.Properties

import org.apache.flink.streaming.api.scala._
import org.apache.flink.streaming.connectors.kafka.{FlinkKafkaConsumer010, FlinkKafkaProducer010}
import org.rawkintrevo.cylon.flinkengine.schemas.KeyedBufferedImageSchema

object Basic {
  def main(args: Array[String]) {

    case class Config(

                       bootStrapServers: String = "localhost:9092",
                       inputTopic: String = "flink",
                       outputTopic: String = "test-flink",
                       droneName: String = "test",
                       parallelism: Int = 1
                     )

    val parser = new scopt.OptionParser[Config]("scopt") {
      head("FlinkEngineDemo", "1.0-SNAPSHOT")

      opt[String]('b', "bootstrapServers").optional()
        .action((x, c) => c.copy(bootStrapServers = x))
        .text("Kafka Bootstrap Servers. Default: localhost:9092")

      opt[String]('i', "inputTopic").optional()
        .action((x, c) => c.copy(inputTopic = x))
        .text("Input Kafka Topic. Default: test")

      opt[String]('o', "outputTopic").optional()
        .action((x, c) => c.copy(outputTopic = x))

      opt[Int]('p', "parallelism").optional()
        .action((x, c) => c.copy(parallelism = 1))

      help("help").text("prints this usage text")
    }

    parser.parse(args, Config()) map { config =>

      val env = StreamExecutionEnvironment.getExecutionEnvironment

      env.setParallelism(config.parallelism) // Changing paralellism increases throughput but really jacks up the video output.

      val properties = new Properties()
      properties.setProperty("bootstrap.servers", config.bootStrapServers)
      properties.setProperty("group.id", "flink")

      val rawVideoConsumer: FlinkKafkaConsumer010[(String, BufferedImage)] = new FlinkKafkaConsumer010[(String, BufferedImage)](config.inputTopic,
        new KeyedBufferedImageSchema(),
        properties)

      rawVideoConsumer.setStartFromLatest()

      val kafkaProducer = new FlinkKafkaProducer010(
        config.outputTopic, // topic
        new KeyedBufferedImageSchema(),
        properties)

      val stream = env
        .addSource(rawVideoConsumer)
        .map(record => {
          val key = record._1
          val image = record._2
          (key, image)
        } )
        .addSink(kafkaProducer)

      // execute program
      env.execute("Flink Cam Markup Engine Demo")

    } getOrElse {
      // arguments are bad, usage message will have been displayed
    }

  }
}





//object FaceDetectorProcessor extends Serializable {
//
////  //System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
////  Class.forName("org.rawkintrevo.cylon.opencv.LoadNative")
////  //NativeUtils.loadOpenCVLibFromJar()
////
////  var inputRawImage: BufferedImage = _
////  var inputMarkupImage: Option[BufferedImage] = _
////  var outputMarkupImage: BufferedImage = _
////
////  var mat: Mat = _
////  //val mat: Mat = bufferedImageToMat(inputRawImage)
////
////  def bufferedImageToMat(bi: BufferedImage): Unit = {
////    // https://stackoverflow.com/questions/14958643/converting-bufferedimage-to-mat-in-opencv
////    mat= new Mat(bi.getHeight, bi.getWidth, CvType.CV_8UC3)
////    val data = bi.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
////    mat.put(0, 0, data)
////
////  }
//
//  var faceRects: Array[MatOfRect] = _
//
//  var faceXmlPaths: Array[String] = _
//  var cascadeColors: Array[Color] = _
//  var cascadeNames: Array[String] = _
//  var faceCascades: Array[CascadeClassifier] = _
//
//  def initCascadeFilters(paths: Array[String], colors: Array[Color], names: Array[String]): Unit = {
//    faceXmlPaths = paths
//    cascadeColors = colors
//    cascadeNames = names
//    faceCascades = faceXmlPaths.map(s => new CascadeClassifier(s))
//  }
//
//  def createFaceRects(): Array[MatOfRect] = {
//
//    var greyMat = new Mat();
//    var equalizedMat = new Mat()
//
//    // Convert matrix to greyscale
//    Imgproc.cvtColor(mat, greyMat, Imgproc.COLOR_RGB2GRAY)
//    // based heavily on https://chimpler.wordpress.com/2014/11/18/playing-with-opencv-in-scala-to-do-face-detection-with-haarcascade-classifier-using-a-webcam/
//    Imgproc.equalizeHist(greyMat, equalizedMat)
//
//    faceRects = (0 until faceCascades.length).map(i => new MatOfRect()).toArray // will hold the rectangles surrounding the detected faces
//
//    for (i <- faceCascades.indices){
//      faceCascades(i).detectMultiScale(equalizedMat, faceRects(i))
//    }
//    faceRects
//  }
//
//  def markupImage(faceRects: Array[MatOfRect]): Unit = {
//
//    val image: BufferedImage = inputMarkupImage match {
//      case img: Some[BufferedImage] => img.get
//      case _ => {
//        val matBuffer = new MatOfByte()
//        Imgcodecs.imencode(".jpg", mat, matBuffer)
//        ImageIO.read(new ByteArrayInputStream(matBuffer.toArray))
//      }
//
//    }
//
//    val graphics = image.getGraphics
//    graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18))
//
//    for (j <- faceRects.indices){
//      graphics.setColor(cascadeColors(j))
//      val name = cascadeNames(j)
//      val faceRectsList = faceRects(j).toList
//      for(i <- 0 until faceRectsList.size()) {
//        val faceRect = faceRectsList.get(i)
//        graphics.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height)
//        graphics.drawString(s"$name", faceRect.x, faceRect.y - 20)
//      }
//    }
//    outputMarkupImage = image
//  }
//
//  def process(image: BufferedImage): BufferedImage = {
//    bufferedImageToMat(image)
//    inputMarkupImage = Some(image)
//    initCascadeFilters(Array("/home/rawkintrevo/gits/opencv/data/haarcascades/haarcascade_profileface.xml",
//      "/home/rawkintrevo/gits/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
//      "/home/rawkintrevo/gits/opencv/data/haarcascades/haarcascade_frontalface_alt.xml",
//      "/home/rawkintrevo/gits/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"),
//      Array(Color.RED, Color.GREEN, Color.BLUE, Color.CYAN),
//      Array("pf", "ff_default", "ff_alt", "ff_alt2")
//    )
//    markupImage(createFaceRects())
//    outputMarkupImage
//  }
//}
