package org.apache.mahout.cylon-example.flinkengine.apps

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

import java.awt.Color
import java.awt.image.BufferedImage
import java.util.Properties

import org.apache.flink.api.common.functions.JoinFunction
import org.apache.flink.api.java.tuple.Tuple
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.streaming.api.TimeCharacteristic
import org.apache.flink.streaming.api.functions.AscendingTimestampExtractor
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator
import org.apache.flink.streaming.api.scala._
import org.apache.flink.streaming.api.scala.function.WindowFunction
import org.apache.flink.streaming.api.windowing.assigners.{SlidingEventTimeWindows, SlidingProcessingTimeWindows, TumblingEventTimeWindows, TumblingProcessingTimeWindows}
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.streaming.connectors.fs.bucketing.BucketingSink
import org.apache.flink.streaming.connectors.kafka.{FlinkKafkaConsumer010, FlinkKafkaProducer010}
import org.apache.flink.util.Collector
import org.apache.mahout.math.algorithms.clustering.CanopyFn
import org.apache.mahout.math.{DenseVector, Matrix, Vector}
import org.apache.mahout.math.algorithms.common.distance.Cosine
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.cylon-example.flinkengine.coprocessfns.MarkupBufferedImagesWithDecomposedFacesCoProcessFunction
import org.apache.mahout.cylon-example.flinkengine.processfns.MergeImagesProcessFunction
import org.apache.mahout.cylon-example.flinkengine.schemas.{KeyedBufferedImageSchema, KeyedFrameBufferedImageSchema, MahoutVectorAndCoordsSchema}
import org.apache.mahout.cylon-example.flinkengine.windowfns._
import org.apache.mahout.cylon-example.frameprocessors.OpenCVImageUtils
import org.slf4j.{Logger, LoggerFactory}

// http://localhost:8090/cylon/cam/test-flink/flink-cluster-cam

object BasicFacialTracking {
  def main(args: Array[String]) {

    var params = ParameterTool.fromArgs(args)


    case class Config(
                       bootStrapServers: String,
                       inputTopic: String,
                       outputTopic: String,
                       parallelism: Int = 1
                     )

      val logger: Logger = LoggerFactory.getLogger(classOf[App])
      val env = StreamExecutionEnvironment.getExecutionEnvironment
      env.getConfig.setGlobalJobParameters(params)

      val config = Config(bootStrapServers = params.has("bootStrapServers") match {
        case true => params.get("bootStrapServers")
        case false => "localhost:9092"
      },
        inputTopic = params.has("inputTopic") match {
          case true => params.get("inputTopic")
          case false => "testTopic"
        },
        outputTopic = params.has("outputTopic") match {
          case true => params.get("outputTopic")
          case false => "test-flink"
        })

      logger.info(s"bootStrapServers: ${config.bootStrapServers}\ninputTopic: ${config.inputTopic}")

      env.setStreamTimeCharacteristic(TimeCharacteristic.ProcessingTime)
      //env.setParallelism(config.parallelism) // Changing paralellism increases throughput but really jacks up the video output.

      val properties = new Properties()
      properties.setProperty("bootstrap.servers", config.bootStrapServers)
      properties.setProperty("group.id", "flink")

      // Load DecomposedFaces
      val vectorAndMetaDataSource: FlinkKafkaConsumer010[(String, Int, Int, Int, Int, Int, Vector)] =
        new FlinkKafkaConsumer010[(String, Int, Int, Int, Int, Int, Vector)](config.inputTopic,
        new MahoutVectorAndCoordsSchema(),
        properties)

      // Load RawImages
      val rawImageSource: FlinkKafkaConsumer010[((String, Int), BufferedImage)] =
        new FlinkKafkaConsumer010[((String, Int), BufferedImage)](config.inputTopic + "-raw_image",
        new KeyedFrameBufferedImageSchema(),
        properties)

      vectorAndMetaDataSource.setStartFromLatest()
      rawImageSource.setStartFromLatest()

      val kafkaProducer = new FlinkKafkaProducer010(
        config.outputTopic, // topic
        new KeyedBufferedImageSchema(),
        properties)


      val vectorAndMetaDataStream: DataStream[DecomposedFace] = env
        .addSource(vectorAndMetaDataSource)
        .map(record => {
          DecomposedFace(s"${record._1}" // key
            , record._2 // h
            , record._3 // w
            , record._4 // x
            , record._5 // y
            , record._6 // frame
            , record._7
            , dvec(record._4.toDouble + (record._3 /2), record._5.toDouble + (record._2 /2), record._6.toDouble))
        } ).name("Vector and Metadata Source")


      /*  This stream generates Canopy Centers- the results are a Matrix which is broadcasted
      *   and can be used to cluster faces as they arrive.
      */
      val canopyStream: DataStream[(String, Matrix)] = vectorAndMetaDataStream
        .assignTimestampsAndWatermarks(new AscendingTimestampExtractor[DecomposedFace] {
          def extractAscendingTimestamp(element: DecomposedFace): Long = System.currentTimeMillis() / 1000L
        })
        .map(face => (face.key, face))
        .keyBy(0)
        .window(SlidingProcessingTimeWindows.of(Time.seconds(2), Time.milliseconds(250)))
        .apply(new CanopyWindowFunction()).name("Fit Canopy Clustering")


    // One stream will be windowed over, say 10 seconds, and identify unique faces (smoothing)
    // E.g. in -> 10 frames, with 17 face rects.  out -> 17 faceRects with "cluster ID" added.
    // val clusteredStream = stream

    val clusteredStream: DataStream[(String, DecomposedFace)] = vectorAndMetaDataStream
      .connect(canopyStream)
      .process(new CanopyAssignmentCoProcessFunction()).name("apply Clusters")
      // Key: String, DecomposedFace, Cluster: Int

    val rawImageStream: DataStream[((String, Int), BufferedImage)] = env
      .addSource(rawImageSource).name("Raw Image Source")

//    rawImageStream.map(t => (t._1._1, t._2))
//      .addSink(kafkaProducer).name(s"http://localhost:8090/cylon/cam/${config.outputTopic}/flink-cluster-cam")


    val imageMarkupStream = clusteredStream
      .connect(rawImageStream)
      .process(new MarkupBufferedImagesWithDecomposedFacesCoProcessFunction(100
                                                      , true)).name("join buffered images")

    val filteredClusterStream = clusteredStream
      .keyBy(0)
      .window(TumblingProcessingTimeWindows.of(Time.seconds(2)))
      .apply(new FilterOutlierClusters(2)).name("filter clusters of less than 2 occurances")


    val filteredImageMarkupStream = filteredClusterStream
      .connect(rawImageStream)
      .process(new MarkupBufferedImagesWithDecomposedFacesCoProcessFunction(100
        , true)).name("join filtered buffered images")


    // execute program

    imageMarkupStream.map(t => (t._1, t._2, 1))
      .union(rawImageStream.map(t => (t._1, t._2, 0)))
      .union(filteredImageMarkupStream.map(t => (t._1, t._2, 3)))
      .process(new MergeImagesProcessFunction(3, "flink-cam"))
      .addSink(kafkaProducer).name("http://localhost:8090/cylon/cam/test-flink/flink-cam")

    env.execute("Flink Basic Facial Tracking Demo")

  }
}
