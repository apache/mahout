package org.apache.mahout.cylon-example.localengine

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

trait AbstractKafkaImageBroadcaster extends AbstractKafkaLocalEngine {


  def writeBufferedImage(topic: String, key: String, img: BufferedImage): Unit = {
    val baos = new ByteArrayOutputStream
    ImageIO.write(img, "jpg", baos)
    writeToKafka(topic, key, baos.toByteArray)
  }

}
