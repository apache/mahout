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

import java.awt.BorderLayout
import javax.swing.JFrame

import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import smile.plot._



/**
  * Create a s surface plot of a DRM by sampling a given percentage
  * and plotting corresponding points of (drmXYZ(::,0), drmXYZ(::,1), drmXYZ(::,2))
  *
  * @param drmXYZ an m x 3 Drm drm to plot
  * @param samplePercent the percentage the drm to sample
  * @tparam K
  */
class MSurf[K](drmXYZ: DrmLike[K], samplePercent: Double = 1, setVisible: Boolean = true) extends MahoutPlot {
  throw new NotImplementedError("This Class is not yet fully implemented.")


  val drmSize = drmXYZ.checkpoint().numRows()
  val sampleDec: Double = samplePercent / 100.toDouble

  val numSamples: Int = (drmSize * sampleDec).toInt

  mPlotMatrix = drmSampleKRows(drmXYZ, numSamples, replacement = false)

  val arrays: Array[Array[Double]]  = Array.ofDim[Double](mPlotMatrix.numRows(), 3)
  for (i <- 0 until mPlotMatrix.numRows()) {
    arrays(i)(0) = mPlotMatrix(i, 0)
    arrays(i)(1) = mPlotMatrix(i, 1)
    arrays(i)(2) = mPlotMatrix(i, 2)
  }

  canvas = Surface.plot(arrays, Palette.jet(256, 1.0f))
  canvas.setTitle("Surface Plot: " + samplePercent + " % sample of " + drmSize +" points")

  plotPanel = new PlotPanel(canvas)

  plotFrame = new JFrame("Surface Plot")
  plotFrame.setLayout(new BorderLayout())
  plotFrame.add(plotPanel)
  plotFrame.setSize(300, 300)
  if (setVisible) {
    plotFrame.setVisible(true)
  }

}
