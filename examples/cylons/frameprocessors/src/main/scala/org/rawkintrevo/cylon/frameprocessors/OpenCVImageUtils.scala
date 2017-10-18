package org.rawkintrevo.cylon.frameprocessors

import java.awt.Font
import java.awt.image.{BufferedImage, DataBufferByte}

import org.opencv.core.{CvType, Mat}
import org.opencv.imgproc.Imgproc


object OpenCVImageUtils {

  Class.forName("org.rawkintrevo.cylon.common.opencv.LoadNative")

  def bufferedImageToMat(bi: BufferedImage): Mat = {
    // https://stackoverflow.com/questions/14958643/converting-bufferedimage-to-mat-in-opencv
    val mat= new Mat(bi.getHeight, bi.getWidth, CvType.CV_8UC3)
    val data = bi.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    mat.put(0, 0, data)
    mat
  }


  def grayAndEqualizeMat(mat: Mat): Mat = {
    var greyMat = new Mat();
    var equalizedMat = new Mat()

    if (mat.channels() > 2) {
      // Convert matrix to greyscale
      Imgproc.cvtColor(mat, greyMat, Imgproc.COLOR_RGB2GRAY)
    } else {
      mat.copyTo(greyMat)
    }

    // based heavily on https://chimpler.wordpress.com/2014/11/18/playing-with-opencv-in-scala-to-do-face-detection-with-haarcascade-classifier-using-a-webcam/
    Imgproc.equalizeHist(greyMat, equalizedMat)
    equalizedMat
  }

  def matToPixelArray(mat: Mat): Array[Double] = {
    val outMat = new Mat()
    mat.convertTo(outMat, CvType.CV_64FC1) // double precision single channel
    val size = (mat.total * mat.channels).asInstanceOf[Int]
    val temp = new Array[Double](size)
    outMat.get(0, 0, temp)
    var i = 0
    while ( {  i < size }) {
      temp(i) = (temp(i) / 2)
      i += 1
    }
    temp


  }

  def bufferedImageToDoubleArray(bi: BufferedImage): Array[Double] = {
    matToPixelArray(
      grayAndEqualizeMat(
        bufferedImageToMat( bi )
      ))
  }

  def drawRectOnImage(img: BufferedImage,
                      x: Int,
                      y: Int,
                      h: Int,
                      w: Int,
                      color: java.awt.Color,
                      caption: String = ""): BufferedImage = {
    BasicImageUtils.drawRectOnImage(img, x, y, h, w, color, caption)
  }
}
