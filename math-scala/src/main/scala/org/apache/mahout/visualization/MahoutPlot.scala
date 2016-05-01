/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package org.apache.mahout.visualization

import java.awt.Graphics2D
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import javax.swing.JFrame

import org.apache.mahout.math.Matrix
import smile.plot.{PlotCanvas, PlotPanel}


trait MahoutPlot  {

  var canvas : PlotCanvas = _
  var plotPanel: PlotPanel = _
  var plotFrame: JFrame = _
  var mPlotMatrix: Matrix = _
  def contentPane = canvas

  // export a PNG of the plot to /tmp/test.png
  def exportPNG(path: String ="/tmp/test.png") = {
    val bi: BufferedImage =
      new BufferedImage(contentPane.getWidth, contentPane.getHeight, BufferedImage.TYPE_INT_ARGB)

    val g2d: Graphics2D = bi.createGraphics

    contentPane.printAll(g2d)

    val file: File = new File(path)

    ImageIO.write(bi, "PNG", file)
  }

}
