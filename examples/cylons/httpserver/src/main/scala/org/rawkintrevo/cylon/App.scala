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

package org.rawkintrevo.cylon

import org.http4s.server.blaze.BlazeBuilder
import org.http4s.util.StreamApp
import org.http4s._
import org.http4s.dsl._
import java.util
import java.util.Properties

import scala.collection.JavaConversions._
import fs2.time
import org.apache.kafka.clients.consumer.KafkaConsumer
import fs2.{Scheduler, Strategy, Task}

import scopt.OptionParser

import scala.concurrent.duration._
import org.slf4j.Logger
import org.slf4j.LoggerFactory


object CylonHTTPServer extends StreamApp {

  def stream(args: List[String]) = {
    case class Config(
                       serverPort: Int = 8080,
                       videoStreamFPS: Int = 4,
                       kafkaBootstrapServer: String = "localhost:9092"
                     )

    var serverPort = 8080
    val parser = new scopt.OptionParser[Config]("scopt") {
      head("CylonCamHTTPServer", "1.0-SNAPSHOT")

      opt[Int]('p', "serverPort").optional()
        .action( (x, c) => c.copy(serverPort = x) )
        .text("Port to serve content on, i.e. 8080 (default)")

      opt[String]('b', "bootstrapServers").optional()
        .action( (x, c) => c.copy(kafkaBootstrapServer = x) )
        .text("Kafka Bootstrap Servers. Default: localhost:9092")

      opt[Int]('f', "fps").optional()
        .action( (x, c) => c.copy(videoStreamFPS = x) )
        .text("Frame Per Second on Video Feeds, Default: 4")

      help("help").text("prints this usage text")
    }

    parser.parse(args, Config()) map { config =>
      CylonHTTPServerObject.fps = config.videoStreamFPS
      CylonHTTPServerObject.bootstrapServer = config.kafkaBootstrapServer
      serverPort = config.serverPort
    } getOrElse {
      // arguments are bad, usage message will have been displayed
    }

    BlazeBuilder.bindHttp(serverPort)
      .mountService(CylonHTTPServerObject.service, "/cylon")
      .serve

  }
}

object CylonHTTPServerObject {

  var fps = 2
  var bootstrapServer = "localhost:9092"
  val milliSecondDelay: Long = 1000 / fps
  val logger: Logger = LoggerFactory.getLogger(classOf[StreamApp])

  implicit val scheduler = Scheduler.fromFixedDaemonPool(2)

  implicit val strategy = Strategy.fromExecutionContext(scala.concurrent.ExecutionContext.Implicits.global)
  // strategy: fs2.Strategy = Strategy
  // An infinite stream of the periodic elapsed time
  val seconds = time.awakeEvery[Task](milliSecondDelay.millisecond)

  val service = HttpService {
    //  We use http4s-dsl to match the path of the Request to the familiar URI form
    case GET -> Root / "cam" / topic / droneName =>
      val props = new Properties()
      props.put("bootstrap.servers", s"$bootstrapServer")
      props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
      props.put("value.deserializer", "org.apache.kafka.common.serialization.ByteArrayDeserializer")
      val id = scala.util.Random.alphanumeric.take(5).mkString("")
      props.put("group.id", s"httpServer_$topic-$id")
      val consumer = new KafkaConsumer[String, Array[Byte]](props)
      consumer.subscribe(util.Collections.singletonList(topic))
      val mediaType: MediaType = MediaType.multipart("x-mixed-replace", boundary = Some("--frame"))
      var lastRelevantImage = new Array[Byte](0)

      Ok().withBody(seconds.map(i => {
        //val record = consumer.poll(milliSecondDelay).records(topic).iterator().toList.last
        for (record <- consumer.poll(milliSecondDelay).records(topic).iterator()) {
          // get the last image in queue that has correct key-value - i.e. drone-name
          if (record.key() == droneName) {
            lastRelevantImage = record.value()
            val imageLength = lastRelevantImage.length
            logger.info(s"Last Image length: ${imageLength}")
            if (imageLength == 0) logger.warn("Record length is 0")

          } else {
            logger.info(s"passed up message with key: ${record.key()}")
          }
        } // for loop
        "--frame\r\nContent-Type: image/jpeg\r\n\r\n".getBytes() ++
          lastRelevantImage ++
          "\r\n\r\n".getBytes()

      })).withContentType(Some(mediaType))
  }
}
