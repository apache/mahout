package org.apache.mahout.cylon-example.frameprocessors

import java.awt.Font
import java.awt.image.BufferedImage

import org.apache.mahout.cylon-example.frameprocessors.OpenCVImageUtils.{bufferedImageToMat, grayAndEqualizeMat, matToPixelArray}

object BasicImageUtils {
  def drawRectOnImage(img: BufferedImage,
                      x: Int,
                      y: Int,
                      h: Int,
                      w: Int,
                      color: java.awt.Color,
                      caption: String = ""): BufferedImage = {
    val graphics = img.getGraphics
    graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18))
    graphics.setColor(color)
    graphics.drawRect(x, y, w, h)
    graphics.drawString(caption, x + w / 2, y + h + 20)
    img
  }

}
