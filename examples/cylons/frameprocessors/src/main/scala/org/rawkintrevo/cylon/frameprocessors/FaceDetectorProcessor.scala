package org.rawkintrevo.cylon.frameprocessors

import java.awt.image.BufferedImage
import java.awt.{Color, Font}
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import org.opencv.core.{Mat, MatOfByte, MatOfRect}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.objdetect.CascadeClassifier

object FaceDetectorProcessor extends Serializable {

  var cc: CascadeClassifier = _

  def initCascadeClassifier(path: String): Unit = {
    cc = new CascadeClassifier(path)
  }

  def createFaceRects(mat: Mat): MatOfRect = {

    val equalizedMat = OpenCVImageUtils.grayAndEqualizeMat(mat)

    val faceRects = new MatOfRect()
    cc.detectMultiScale(equalizedMat, faceRects)

    faceRects
  }
}
//  def getFaces(mat: Mat): Array[Mat] ={
//    val faceRects = createFaceRects(mat)
//
//    for (rect <- faceRects.toArray){
//      rect.x, rect.y, rect.
//    }

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
//    mat = ImageUtils.bufferedImageToMat(image)
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

