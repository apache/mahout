package org.rawkintrevo.cylon.flinkengine.processfns

import java.awt.image.BufferedImage
import java.awt.Color


import org.apache.flink.streaming.api.functions.ProcessFunction
import org.apache.flink.util.Collector

class MergeImagesProcessFunction(numOfImages: Int,
                                 outputKey: String)
  extends ProcessFunction[((String,Int), BufferedImage, Int), (String, BufferedImage) ] {

  val bufferedImages: scala.collection.mutable.Map[(String, Int), scala.collection.mutable.Map[Int, BufferedImage]]
    = scala.collection.mutable.Map()

  var minFrame = 0
  var minImages = 0
  override def processElement(i: ((String, Int), BufferedImage, Int),
                              context: ProcessFunction[((String, Int), BufferedImage, Int), (String, BufferedImage)]#Context,
                              collector: Collector[(String, BufferedImage)]): Unit = {
    if (!bufferedImages.contains(i._1)){
      bufferedImages(i._1) = scala.collection.mutable.Map[Int, BufferedImage]()
    }
    bufferedImages(i._1)(i._3) = i._2

    if (bufferedImages(i._1).keys.size > minImages){
      minImages = bufferedImages(i._1).keys.size
    }

    if (bufferedImages(i._1).keys.size == numOfImages) {//== numOfImages) {
      val outImage = joinBufferedImage(bufferedImages(i._1).toMap)
      //bufferedImages.remove(i._1)

      minFrame = i._1._2
      collector.collect((outputKey, outImage))
    }
  }



  def joinBufferedImage(imgMap: Map[Int, BufferedImage]): BufferedImage = { //do some calculate first
    val maxWide = 4
    val offset = 5
    val wid = (imgMap(0).getWidth + offset) * numOfImages // All images need to be same size or things get wierd
    val height = imgMap(0).getHeight //(imgArray(0).getHeight + offset) * imgArray.length
    // I'll wrap this later- for now just side-by
    //create a new buffer and draw two image into the new image
    val newImage = new BufferedImage(wid, height, BufferedImage.TYPE_INT_ARGB)
    val g2 = newImage.createGraphics
    val oldColor = g2.getColor
    //fill background
    g2.setPaint(Color.WHITE)
    g2.fillRect(0, 0, wid, height)
    //draw image
    g2.setColor(oldColor)

    for (k <- imgMap.keys.toArray.sorted){
      g2.drawImage(imgMap(k), null, imgMap(k).getWidth * k + offset, 0)
    }
    g2.dispose()
    newImage
  }
}
