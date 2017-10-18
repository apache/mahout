package org.rawkintrevo.cylon.examples.localfacialrecognition

import java.awt.{Color, Font}
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.ByteArrayInputStream
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import javax.imageio.ImageIO

import org.apache.mahout.math.{DenseVector, Vector}
import org.apache.solr.common.{SolrDocument, SolrInputDocument}
import org.opencv.core._
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.opencv.videoio.VideoCapture
import org.rawkintrevo.cylon.common.mahout.MahoutUtils
import org.rawkintrevo.cylon.common.solr.CylonSolrClient
import org.rawkintrevo.cylon.frameprocessors.{FaceDetectorProcessor, OpenCVImageUtils}
import org.rawkintrevo.cylon.localengine.{AbstractKafkaImageBroadcaster, KafkaFaceDecomposer}

class SimpleKafkaFacialRecognition(topic: String, key: String)
  extends KafkaFaceDecomposer(topic: String, key: String)
  with AbstractKafkaImageBroadcaster {

  var threshold: Double = 2000.0


  override def run(): Unit = {
    Class.forName("org.rawkintrevo.cylon.common.opencv.LoadNative")


    val videoCapture = new VideoCapture
    logger.info(s"Attempting to open video source at ${inputPath}")
    videoCapture.open(inputPath)

    if (!videoCapture.isOpened) logger.warn("Camera Error")
    else logger.info(s"Successfully opened video source at ${inputPath}")

    // Create Cascade Filter /////////////////////////////////////////////////////////////////////////////////////////
    FaceDetectorProcessor.initCascadeClassifier(cascadeFilterPath)

    // Init variables needed /////////////////////////////////////////////////////////////////////////////////////////
    var mat = new Mat()

    val solrClient = cylonSolrClient.solrClient
    var facesInView = 0

    var lastRecognizedHuman = ""
    var stateCounter = new Array[Int](5)


    while (videoCapture.read(mat)) {



      val faceRects = FaceDetectorProcessor.createFaceRects(mat)

      val faceArray = faceRects.toArray

      // Scale faces to 250x250 and convert to Mahout DenseVector
      val faceVecArray: Array[DenseVector] = faceArray.map(r => {
        val faceMat = new Mat(mat, r)
        val size: Size = new Size(250, 250)
        val resizeMat = new Mat(size, faceMat.`type`())
        Imgproc.resize(faceMat, resizeMat, size)
        val faceVec = new DenseVector(OpenCVImageUtils.matToPixelArray(OpenCVImageUtils.grayAndEqualizeMat(resizeMat)))
        faceVec
      })

      // Decompose Image into linear combo of eigenfaces (which were calulated offline)
      val faceDecompVecArray: Array[Vector] = faceVecArray
        .map(v => MahoutUtils.decomposeImgVecWithEigenfaces(v.minus(colCentersV), eigenfacesInCore))


//      for (vec <- faceDecompVecArray) {
//        writeOutput(vec)
//      }

      val nFaces = faceRects.toArray.length

      var triggerDeltaFaces = false
      // Has there been a change in the number of faces in view?
      if (nFaces > facesInView) {
        // we have a new face(s) execute code
        logger.debug(s"where once there were $facesInView, now there are $nFaces")
        triggerDeltaFaces = true
      } else if (nFaces < facesInView) {
        // someone left the frame
        facesInView = nFaces
        triggerDeltaFaces = false
      } else {
        // things to do in the condition of no change
        triggerDeltaFaces = false
      }

      if (triggerDeltaFaces) {
        val faceArray = faceRects.toArray

        // Scale faces to 250x250 and convert to Mahout DenseVector
        val faceVecArray: Array[DenseVector] = faceArray.map(r => {
          val faceMat = new Mat(mat, r)
          val size: Size = new Size(250, 250)
          val resizeMat = new Mat(size, faceMat.`type`())
          Imgproc.resize(faceMat, resizeMat, size)
          val faceVec = new DenseVector(OpenCVImageUtils.matToPixelArray(OpenCVImageUtils.grayAndEqualizeMat(resizeMat)))
          faceVec
        })

        // Decompose Image into linear combo of eigenfaces (which were calulated offline)
        // todo: don't forget to meanCenter faceVects
        val faceDecompVecArray = faceVecArray.map(v => MahoutUtils.decomposeImgVecWithEigenfaces(v, eigenfacesInCore))
        // drop first 3 elements as they represent 3 dimensional light

        // Query Solr
        import org.apache.solr.client.solrj.SolrQuery
        import org.apache.solr.client.solrj.SolrQuery.SortClause
        import org.apache.mahout.math.scalabindings.MahoutCollections._
        import org.apache.solr.client.solrj.response.QueryResponse

        def eigenFaceQuery(v: org.apache.mahout.math.Vector): QueryResponse = {
          val query = new SolrQuery
          query.setRequestHandler("/select")
          val currentPointStr = v.toArray.mkString(",")
          val eigenfaceFieldNames = (0 until faceDecompVecArray(0).size()).map(i => s"e${i}_d").mkString(",")
          val distFnStr = s"dist(2, ${eigenfaceFieldNames},${currentPointStr})"
          query.setQuery("*:*")
          query.setSort(new SortClause(distFnStr, SolrQuery.ORDER.asc))
          query.setFields("name_s", "calc_dist:" + distFnStr, "last_seen_pdt")
          query.setRows(10)


          val response: QueryResponse = solrClient.query(query)
          response
        }

        def insertNewFaceToSolr(v: org.apache.mahout.math.Vector) = {
          val doc = new SolrInputDocument()
          val humanName = "human-" + scala.util.Random.alphanumeric.take(5).mkString("").toUpperCase
          logger.info(s"I think I'll call you '$humanName'")
          doc.addField("name_s", humanName)
          doc.addField("last_seen_pdt", ZonedDateTime.now.format(DateTimeFormatter.ISO_INSTANT)) // YYYY-MM-DDThh:mm:ssZ   DateTimeFormatter.ISO_INSTANT, ISO-8601
          v.toMap.map { case (k, v) => doc.addField(s"e${k.toString}_d", v) }
          solrClient.add(doc)
          logger.debug("Flushing new docs to solr")
          solrClient.commit()
          humanName
        }

        def getDocsArray(response: QueryResponse): Array[SolrDocument] = {
          val a = new Array[SolrDocument](response.getResults.size())
          for (i <- 0 until response.getResults.size()) {
            a(i) = response.getResults.get(i)
          }
          a
        }

        def lastRecognizedHumanStillPresent(response: QueryResponse): Boolean ={
          val a = getDocsArray(response)
          a.exists(_.get("name_s") == lastRecognizedHuman)
        }

        def lastRecognizedHumanDistance(response: QueryResponse): Double ={
          val a = getDocsArray(response)
          var output: Double = 1000000000
          if (lastRecognizedHumanStillPresent(response)) {
            output = a.filter(_.get("name_s") == lastRecognizedHuman)(0).get("calc_dist").asInstanceOf[Double]
          }
          output
        }


        // todo: replace faceDecompVecArray(0) with for function and iterate
        val response = eigenFaceQuery(faceDecompVecArray(0))

        // in essence canopy clustering....
        /**
          * // (1) Orig Point: new center
          * If next dist(p1, p2) < d1 -> Same Point
          * If next dist(p1, p2) < d2 -> maybe same point
          * Else -> new center
          */
        // Need a buffer- e.g. needs to be outside looseTolerance for n-frames
        val tightTolerance = 1500
        val looseTolerance = 5500
        val minFramesInState = 10

        // (1)
        if (response.getResults.size() == 0) {
          logger.info("I'm a stupid baby- everyone is new to me.")
          insertNewFaceToSolr(faceDecompVecArray(0))
        }

        if (response.getResults.size() > 0) {
          val bestName: String = response.getResults.get(0).get("name_s").asInstanceOf[String]
          val bestDist = response.getResults.get(0).get("calc_dist").asInstanceOf[Double]
          if (lastRecognizedHuman.equals("")) {
            lastRecognizedHuman = bestName
          }
          if (lastRecognizedHumanDistance(response) < tightTolerance){   // State 0
            stateCounter = new Array[Int](5)
            logger.info(s"still $bestName")
          } else if (lastRecognizedHumanDistance(response) < looseTolerance) { // State 1
            stateCounter(1) += 1
            logger.info(s"looks like $bestName")
          } else if (bestDist < tightTolerance) {  // State 2
            if (stateCounter(2) > minFramesInState){
              lastRecognizedHuman = bestName
              logger.info(s"oh hai $bestName ")
              stateCounter = new Array[Int](5)
            } else {stateCounter(2) += 1}
          } else if (bestDist < looseTolerance) {  // State 3
            stateCounter(3) += 1
            logger.info(s"oh god it looks like $bestName, must be the clouds in eyes... $lastRecognizedHuman - ${lastRecognizedHumanDistance(response)} vs $bestDist")
          } else { // State 4
            if (stateCounter(4) > minFramesInState) {
              lastRecognizedHuman = insertNewFaceToSolr(faceDecompVecArray(0))
              stateCounter = new Array[Int](5)
            } else {
              //logger.info(s"no idea, but its been ${stateCounter(4)} frames")
              stateCounter(4) += 1
            }
            // logging handled in subcall

          }

          println(s"$lastRecognizedHuman " + stateCounter.mkString(","))

        }

        if (faceArray.length > 0){
          FaceDetectorProcessor_v1.mat = mat
          FaceDetectorProcessor_v1.initCascadeFilters(Array(""), Array(Color.GREEN), Array(lastRecognizedHuman))
          FaceDetectorProcessor_v1.markupImage(Array(new MatOfRect(faceRects.rowRange(0,1))))
          writeBufferedImage(topic, key, FaceDetectorProcessor_v1.outputMarkupImage)
        }

        if (writeBufferedImages) {
          val matBuffer = new MatOfByte()
          Imgcodecs.imencode(".jpg", mat, matBuffer)
          val img: BufferedImage = ImageIO.read(new ByteArrayInputStream(matBuffer.toArray))
          writeBufferedImage(topic, key + "_raw_image", img)
        }
      }
    }

  }
}


object FaceDetectorProcessor_v1 extends Serializable {

  // ** Lifting this old code to make a demo work quick and dirty -will refactor later

  //System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
  Class.forName("org.rawkintrevo.cylon.common.opencv.LoadNative")
  //NativeUtils.loadOpenCVLibFromJar()

  var inputRawImage: BufferedImage = _
  var inputMarkupImage: Option[BufferedImage] = _
  var outputMarkupImage: BufferedImage = _

  var mat: Mat = _
  //val mat: Mat = bufferedImageToMat(inputRawImage)

  def bufferedImageToMat(bi: BufferedImage): Unit = {
    // https://stackoverflow.com/questions/14958643/converting-bufferedimage-to-mat-in-opencv
    mat= new Mat(bi.getHeight, bi.getWidth, CvType.CV_8UC3)
    val data = bi.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    mat.put(0, 0, data)

  }

  var faceRects: Array[MatOfRect] = _

  var faceXmlPaths: Array[String] = _
  var cascadeColors: Array[Color] = _
  var cascadeNames: Array[String] = _
  var faceCascades: Array[CascadeClassifier] = _

  def initCascadeFilters(paths: Array[String], colors: Array[Color], names: Array[String]): Unit = {
    faceXmlPaths = paths
    cascadeColors = colors
    cascadeNames = names
    //faceCascades = faceXmlPaths.map(s => new CascadeClassifier(s))
    // disabled bc I'm just using this object to render some shit for a demo
  }

  def createFaceRects(): Array[MatOfRect] = {

    var greyMat = new Mat();
    var equalizedMat = new Mat()

    // Convert matrix to greyscale
    Imgproc.cvtColor(mat, greyMat, Imgproc.COLOR_RGB2GRAY)
    // based heavily on https://chimpler.wordpress.com/2014/11/18/playing-with-opencv-in-scala-to-do-face-detection-with-haarcascade-classifier-using-a-webcam/
    Imgproc.equalizeHist(greyMat, equalizedMat)

    faceRects = (0 until faceCascades.length).map(i => new MatOfRect()).toArray // will hold the rectangles surrounding the detected faces

    for (i <- faceCascades.indices){
      faceCascades(i).detectMultiScale(equalizedMat, faceRects(i))
    }
    faceRects
  }

  def markupImage(faceRects: Array[MatOfRect]): Unit = {

    val image: BufferedImage = inputMarkupImage match {
      case img: Some[BufferedImage] => img.get
      case _ => {
        val matBuffer = new MatOfByte()
        Imgcodecs.imencode(".jpg", mat, matBuffer)
        ImageIO.read(new ByteArrayInputStream(matBuffer.toArray))
      }

    }

    val graphics = image.getGraphics
    graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18))

    for (j <- faceRects.indices){
      graphics.setColor(cascadeColors(j))
      val name = cascadeNames(j)
      val faceRectsList = faceRects(j).toList
      for(i <- 0 until faceRectsList.size()) {
        val faceRect = faceRectsList.get(i)
        graphics.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height)
        graphics.drawString(s"$name", faceRect.x, faceRect.y - 20)
      }
    }
    outputMarkupImage = image
  }

  def process(image: BufferedImage): BufferedImage = {
    bufferedImageToMat(image)
    inputMarkupImage = Some(image)
    initCascadeFilters(Array("/home/rawkintrevo/gits/opencv/data/haarcascades/haarcascade_profileface.xml",
      "/home/rawkintrevo/gits/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
      "/home/rawkintrevo/gits/opencv/data/haarcascades/haarcascade_frontalface_alt.xml",
      "/home/rawkintrevo/gits/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"),
      Array(Color.RED, Color.GREEN, Color.BLUE, Color.CYAN),
      Array("pf", "ff_default", "ff_alt", "ff_alt2")
    )
    markupImage(createFaceRects())
    outputMarkupImage
  }
}