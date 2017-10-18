package org.rawkintrevo.cylon.frameprocessors

import java.awt.image.{BufferedImage, DataBufferByte}

import org.opencv.core.{CvType, Mat}


trait FrameProcessor extends Serializable {

  Class.forName("org.rawkintrevo.cylon.opencv.LoadNative")


  var inputRawImage: BufferedImage = _
  var inputMarkupImage: Option[BufferedImage] = _
  var outputMarkupImage: BufferedImage = _

  var mat: Mat = _
  //val mat: Mat = bufferedImageToMat(inputRawImage)

}