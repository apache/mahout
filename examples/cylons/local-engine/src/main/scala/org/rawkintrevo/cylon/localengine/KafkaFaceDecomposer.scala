package org.apache.mahout.cylon-example.localengine

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import org.apache.mahout.math.Vector
import org.opencv.core.{Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs
import org.apache.mahout.cylon-example.common.mahout.MahoutUtils
import org.apache.mahout.cylon-example.frameprocessors.FaceDetectorDemo.mat


class KafkaFaceDecomposer(topic: String, key: String)
  extends AbstractKafkaLocalEngine
    with AbstractFaceDecomposer
    with AbstractKafkaImageBroadcaster{

  var writeBufferedImages: Boolean = false

  def writeOutput(vec: Vector) = {
    writeToKafka(topic, key, MahoutUtils.vector2byteArray(vec))
  }

  override def otherMatOps(mat: Mat, frame: Int): Unit = {
    if (writeBufferedImages) {

      val matBuffer = new MatOfByte()
      Imgcodecs.imencode(".jpg", mat, matBuffer)
      val img: BufferedImage = ImageIO.read(new ByteArrayInputStream(matBuffer.toArray))
      val mod_key = s"${key}-${frame}"
      //logger.debug(s"writing frame $frame to kafka topic: $topic with key $mod_key with message as ${img.getHeight()} x ${img.getWidth()} image")
      writeBufferedImage(topic+"-raw_image", mod_key, img)
    }
  }

}


