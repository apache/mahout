package org.apache.mahout.cylon-example.frameprocessors

import java.awt.image.{BufferedImage, DataBufferByte}

import org.opencv.core.{CvType, Mat}


trait FrameProcessor extends Serializable {

  Class.forName("org.apache.mahout.cylon-example.opencv.LoadNative")


  var inputRawImage: BufferedImage = _
  var inputMarkupImage: Option[BufferedImage] = _
  var outputMarkupImage: BufferedImage = _

  var mat: Mat = _
  //val mat: Mat = bufferedImageToMat(inputRawImage)

}