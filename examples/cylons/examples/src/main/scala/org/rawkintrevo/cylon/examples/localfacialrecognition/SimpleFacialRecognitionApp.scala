package org.rawkintrevo.cylon.examples.localfacialrecognition

import org.rawkintrevo.cylon.localengine.KafkaFaceDecomposer
import org.slf4j.{Logger, LoggerFactory}

/**
 * @author ${user.name}
 */
object SimpleFacialRecognitionApp {
  val logger: Logger = LoggerFactory.getLogger(classOf[App])

  def main(args : Array[String]) {

    case class Config(
                       eigenfacesPath: String = "",
                       colMeansPath: String = "",
                       solrURL: String = "",
                       videoStreamURL: String = "rtsp://192.168.100.1:554/cam1/mpeg4",
                       cascadeFilterPath: String = "",
                       distanceTolerance: Double = 2000.0,
                       rawImages: Boolean = false
                     )


    val parser = new scopt.OptionParser[Config]("scopt") {
      head("Local Engine", "1.0-SNAPSHOT")

      opt[String]('c', "cascadeFilterPath").required()
        .action((x, c) => c.copy(cascadeFilterPath = x))
        .text("Path to OpenCV Cascade Filter to use, e.g. $OPENCV_3_0/data/haarcascades/haarcascade_frontalface_alt.xml")

      opt[String]('e', "eigenfacesPath").required()
        .action((x, c) => c.copy(eigenfacesPath = x))
        .text("Path to output of eigenfaces file, e.g. $CYLON_HOME/data/eigenfaces.mmat")

      opt[String]('m', "colCentersPath").required()
        .action((x, c) => c.copy(colMeansPath = x))
        .text("Path to output of col centers file that was generated with eigenfaces file, e.g. $CYLON_HOME/data/colMeans.mmat")

      opt[String]('i', "inputVideoURL").optional()
        .action((x, c) => c.copy(videoStreamURL = x))
        .text("URL of input video, use 'http://bglive-a.bitgravity.com/ndtv/247hi/live/native' for testing, defaults to 'rtsp://192.168.100.1:554/cam1/mpeg4' (drone cam address)")

      opt[String]('s', "solrURL").required()
        .action((x, c) => c.copy(solrURL = x))
        .text("URL of Solr, e.g. http://localhost:8983/solr/cylonfaces")

      opt[Boolean]('r', "rawImages").optional()
        .action((x, c) => c.copy(rawImages = x))
        .text("Emit a stream of raw images (usually for demo purposes).  Topic will be  assigned topic + '_raw_images', eg. test_raw_images")


      help("help").text("prints this usage text")
    }

    parser.parse(args, Config()) map { config =>
      logger.info("Local Engine Started")

      val engine = new SimpleKafkaFacialRecognition("test", "test")
      engine.connectToSolr(config.solrURL)
      engine.loadEigenFacesAndColCenters(config.eigenfacesPath, config.colMeansPath)
      engine.cascadeFilterPath = config.cascadeFilterPath
      engine.setupKafkaProducer()
      engine.inputPath = config.videoStreamURL
      engine.writeBufferedImages = config.rawImages

      engine.run()
      sys.exit(1)
    } getOrElse{

    }
  }
}
