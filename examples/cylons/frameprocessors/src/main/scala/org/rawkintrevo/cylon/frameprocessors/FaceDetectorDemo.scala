package org.rawkintrevo.cylon.frameprocessors

import java.awt.{Color, Font}

import org.opencv.core.{ Mat, MatOfByte, MatOfRect}
import java.awt.image.{BufferedImage}
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier

object FaceDetectorDemo extends FrameProcessor with Serializable {
  var faceRects: Array[MatOfRect] = _

  var faceXmlPaths: Array[String] = _
  var cascadeColors: Array[Color] = _
  var cascadeNames: Array[String] = _
  var faceCascades: Array[CascadeClassifier] = _

  def initCascadeFilters(paths: Array[String], colors: Array[Color], names: Array[String]): Unit = {
    faceXmlPaths = paths
    cascadeColors = colors
    cascadeNames = names
    faceCascades = faceXmlPaths.map(s => new CascadeClassifier(s))
  }

  def createFaceRects(): Array[MatOfRect] = {

    val equalizedMat = OpenCVImageUtils.grayAndEqualizeMat(mat)

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
    mat = OpenCVImageUtils.bufferedImageToMat(image)
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
