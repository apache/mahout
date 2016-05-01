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

import java.awt.{BorderLayout, Color}
import javax.swing.JFrame

import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import smile.plot._


/**
  * Create a Histogram of bims of a DRM by sampling a given percentage
  * and plotting corresponding points of (drmXY(::,0),drmXY(::,1))
  *
  * @param drmXY an m x 1 Drm Column, drm to plot
  * @param numBins: number of bins
  * @param samplePercent the percentage the drm to sample. Default =1
  * @tparam K
  */
class MHisto[K](drmXY: DrmLike[K], numBins: Int, samplePercent: Double = 1, setVisible: Boolean = true)  extends MahoutPlot {
  val drmSize = drmXY.checkpoint().numRows()
  val sampleDec: Double = (samplePercent / 100.toDouble)

  val numSamples: Int = (drmSize * sampleDec).toInt

  mPlotMatrix = drmSampleKRows(drmXY, numSamples, false)
  val arrays = Array.ofDim[Double](mPlotMatrix.numRows())
  for (i <- 0 until mPlotMatrix.numRows()) {
       arrays(i) = mPlotMatrix(i, 0)
  }

  // just use bins during development, can define ranges etc later
  canvas = Histogram.plot(arrays, numBins)
  canvas.setTitle("2d Histogram: " + samplePercent + " % sample of " + drmSize +" points")
  canvas.setAxisLabels("x_0", "frequency")

  plotPanel = new PlotPanel(canvas)

  plotFrame = new JFrame("2d Histogram")
  plotFrame.setLayout(new BorderLayout())
  plotFrame.add(plotPanel)
  plotFrame.setSize(300,300)
  if (setVisible) {
     plotFrame.setVisible(true)
  }

}
