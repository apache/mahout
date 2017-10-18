package org.rawkintrevo.cylon.flinkengine.coprocessfns

import java.awt.image.BufferedImage
import java.awt.{Color, Font}

import org.apache.flink.streaming.api.functions.co.CoProcessFunction
import org.apache.flink.util.Collector
import org.rawkintrevo.cylon.flinkengine.windowfns.DecomposedFace

/**
  * pe1: When a Decomposed Face Comes in add it to the appropriate frame
  *
  * pe2: When an Image comes in, save it to a Map (so that it can be looked up easily by the pe1
  */
class MarkupBufferedImagesWithDecomposedFacesCoProcessFunction(frameDelayInit: Int = 100,
                                                               adaptiveFrameDelay: Boolean = false,
                                                               drawRects: Boolean = true,
                                                               drawClusters: Boolean = true,
                                                               drawName: Boolean = true) extends CoProcessFunction[
  (String, DecomposedFace),
  ((String, Int), BufferedImage),
  ((String, Int), BufferedImage)] {

  val bufferedImages: scala.collection.mutable.Map[(String, Int), BufferedImage] = scala.collection.mutable.Map()

  var frameDelay = frameDelayInit
  var frameInterval = 0

  var lastIn1Frame = 0
  var lastIn2Frame = 0
  var lastEmittedFrame = 0

  var textBaseHeight = 0
  var lastLostRecFrame = 0
  var lastLostRecLag = 0
  var lostRectsCount = 0

  var adaptiveCounter = 0
  var framesBetweenAdaptiveFrameDelayDecrease = 20

  def processElement1(in1: (String, DecomposedFace),
    context: CoProcessFunction[(String, DecomposedFace), ((String, Int), BufferedImage), ((String, Int), BufferedImage)]#Context,
    collector: Collector[((String, Int), BufferedImage)] ): Unit ={
    // Processes Face Vects
    val key = in1._1
    val decomposedFace = in1._2
    var cluster = decomposedFace.cluster
    frameInterval = Math.max(decomposedFace.frame - lastIn1Frame, frameInterval) // don't go to zero when we get multiple rects on same frame
    lastIn1Frame = decomposedFace.frame

    if (!bufferedImages.contains((decomposedFace.key, decomposedFace.frame))) {
      lostRectsCount += 1
      lastLostRecFrame = decomposedFace.frame
      lastLostRecLag = lastIn2Frame - lastLostRecFrame
      return
    }

    val image: BufferedImage = bufferedImages((decomposedFace.key, decomposedFace.frame))
    val graphics = image.getGraphics
    val x = decomposedFace.x
    val y = decomposedFace.y
    val w = decomposedFace.w
    val h = decomposedFace.h

    val colors = Map(
      -1 -> Color.GRAY,
      0 -> Color.BLUE,
      1 -> Color.RED,
      2 -> Color.GREEN,
      3 -> Color.WHITE,
      4 -> Color.DARK_GRAY,
      5 -> Color.CYAN,
      6 -> Color.MAGENTA,
      7 -> Color.YELLOW,
      8 -> Color.ORANGE,
      9 -> Color.PINK
    )
    if (drawName && decomposedFace.name == "ghost"){
      graphics.setColor(Color.LIGHT_GRAY)
    } else {
      graphics.setColor(colors(cluster % 10))
    }


    if (drawRects) {
      graphics.drawRect(x, y, w, h)
    }

    if (drawClusters) {
      var clusterText = s"cluster-$cluster-${decomposedFace.distanceFromCenter.round}"
      //if (cluster == -1) clusterText = "unclustered"

      graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 10))
      graphics.drawString(clusterText, x + w / 2, y + h - 20)
    }

    if (drawName) {
      graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18))
      graphics.drawString(decomposedFace.name, x + w / 2, y + h)
    }

    // Put updated image back in the buffer- this was just one rect.
    bufferedImages((decomposedFace.key, decomposedFace.frame)) = image // outImg



  }

  override def processElement2(in2: ((String, Int), BufferedImage),
                               context: CoProcessFunction[(String, DecomposedFace), ((String, Int), BufferedImage), ((String, Int), BufferedImage)]#Context,
                               collector: Collector[((String,Int), BufferedImage)]): Unit = {

    val frame = in2._1._2
    val key = in2._1._1
    val img = in2._2
    val newImage = new BufferedImage(img.getWidth + 100, img.getHeight+50, BufferedImage.TYPE_INT_ARGB)
    val graphics = newImage.createGraphics
    graphics.setColor(Color.BLACK)
    graphics.fillRect(0, 0, newImage.getWidth, newImage.getHeight)
    graphics.drawImage(img, null, 0, 0)
    textBaseHeight = img.getHeight
    graphics.setFont(new Font(Font.SERIF, Font.PLAIN, 10))
    graphics.setColor(Color.green)
    graphics.drawString(s"Frame: $frame Key: $key", 10, textBaseHeight + 11)


    bufferedImages((key, frame)) = newImage
    // need to remove fired image from buffer...
    // need to add some sort of adaptive mechanism for autoTuning the frame delay.

    // Gotta let it heat up, ow "key not found errors"
    if (bufferedImages.contains((key, frame - frameDelay))){
      val emitImage = bufferedImages((key, frame - frameDelay))
      val graphics = emitImage.getGraphics
      graphics.setFont(new Font(Font.SERIF, Font.PLAIN, 10))
      graphics.setColor(Color.green)
      graphics.drawString(s"LastIn1Frame: $lastIn1Frame LostRects: $lostRectsCount  LastLostRecFrame: $lastLostRecFrame", 10, textBaseHeight + 22)
      graphics.drawString(s"LastLostRecLag: $lastLostRecLag FrameDelay: $frameDelay", 10, textBaseHeight + 33)

      // Very Stupid Throttling (that can only increase)
      //frameDelay = Math.max(lastLostRecLag, frameDelay)

      collector.collect((in2._1, emitImage))
    }

    // Clean old buffered Images
//    bufferedImages.keys.toArray
//      .filter(_._2 < in2._1._2 - frameDelay * 2)
//      .map(k => bufferedImages.remove((k._1, k._2)))

  }
}
