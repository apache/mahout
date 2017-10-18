package org.rawkintrevo.cylon.localengine

import org.apache.mahout.math.{DenseVector, Matrix, Vector}
import org.apache.mahout.math._
import org.apache.mahout.math.algorithms.preprocessing.MeanCenter
import org.apache.mahout.math.decompositions._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.scalabindings.MahoutCollections._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.opencv.core.{Core, Mat, Size}
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.{VideoCapture, Videoio}
import org.rawkintrevo.cylon.common.mahout.MahoutUtils
import org.rawkintrevo.cylon.frameprocessors.{FaceDetectorProcessor, OpenCVImageUtils}

trait AbstractFaceDecomposer extends AbstractLocalEngine {

  var eigenfacesInCore: Matrix = _
  var colCentersV: Vector = _
  var cascadeFilterPath: String = _
  var fastForward: Boolean = true

  var targetFps: Double = 5.0

  var includeMeta: Boolean = false

  def writeOutput(vec: Vector)

  def loadEigenFacesAndColCenters(efPath: String, ccPath: String): Unit = {
    logger.info(s"Loading Eigenfaces from ${efPath}")
    eigenfacesInCore = MahoutUtils.matrixReader(efPath)
    val efRows = eigenfacesInCore.numRows()
    val efCols = eigenfacesInCore.numCols()
    logger.info(s"Loaded Eigenfaces matrix ${efRows}x${efCols}")

    colCentersV = MahoutUtils.vectorReader(ccPath)
  }

  def run() = {
    //System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    Class.forName("org.rawkintrevo.cylon.common.opencv.LoadNative")

    val videoCapture = new VideoCapture
    logger.info(s"Attempting to open video source at ${inputPath}")
    videoCapture.open(inputPath)

    if (!videoCapture.isOpened) logger.warn("Camera Error")
    else logger.info(s"Successfully opened video source at ${inputPath}")

    logger.info(s"Input Video FPS: ${videoCapture.get(Videoio.CAP_PROP_FPS)}")
    logger.info(s"Desired FPS: ${targetFps}")
    val capEveryNthFrame = Math.min((videoCapture.get(Videoio.CAP_PROP_FPS) / targetFps).round.toInt, 1)
    logger.info(s"capEveryNthFrame: ${capEveryNthFrame}")
    // Create Cascade Filter /////////////////////////////////////////////////////////////////////////////////////////
    FaceDetectorProcessor.initCascadeClassifier(cascadeFilterPath)

    // Init variables needed /////////////////////////////////////////////////////////////////////////////////////////
    var mat = new Mat()

    var frame: Int = 0

    while (videoCapture.read(mat)) {
      frame += 1
      if (frame % capEveryNthFrame == 0) {
        writeFaceRects(mat, frame)
        otherMatOps(mat, frame)

        if (fastForward) {
          // https://stackoverflow.com/questions/21066875/opencv-constants-captureproperty
          // this doesn't appear to work for the indian TV test video
          // val CV_CAP_PROP_FRAME_COUNT =7
          val ff_frame: Double = videoCapture.get(Videoio.CAP_PROP_FRAME_COUNT)
//          logger.debug(s"Most rescent frame (from OpenCV): $ff_frame")
//          logger.debug(s"local counter reports: $frame")
//          logger.debug(s"current frame (from OpenCV): ${videoCapture.get(Videoio.CAP_PROP_POS_FRAMES)}")
          //val CV_CAP_PROP_POS_FRAMES     =1
          // Fast Forward to Latest Frame
          videoCapture.set(Videoio.CAP_PROP_POS_FRAMES, ff_frame - 1)
        }
      }
    }

  }

  def fastForwardVideo() = {

  }

  def writeFaceRects(mat: Mat, frame: Int): Unit = {

    val faceRects = FaceDetectorProcessor.createFaceRects(mat)

    logger.debug(s"detected ${faceRects.rows()} faces")

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
    var faceDecompVecArray: Array[Vector] = faceVecArray
      .map(v => MahoutUtils.decomposeImgVecWithEigenfaces(v.minus(colCentersV), eigenfacesInCore))

    if (includeMeta) {
      val metaArray: Array[Array[Double]] = faceArray.map(r => Array(r.height.toDouble,
        r.width.toDouble,
        r.x.toDouble,
        r.y.toDouble,
        frame.toDouble))

      for (i <- faceDecompVecArray.indices){
        val tmp: Array[Double] = faceDecompVecArray(i).toArray
        val tmp2: Array[Double] = metaArray(i) ++ tmp
        faceDecompVecArray(i) = dvec( tmp2 )
      }
    }

    logger.info(s"frame: $frame ,writing ${faceDecompVecArray.length} face vectors")
    for (vec <- faceDecompVecArray) {

      writeOutput(vec)
    }
  }

  def otherMatOps(mat: Mat, frame: Int): Unit = {

  }
}
