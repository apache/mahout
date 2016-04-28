package org.apache.mahout.visiualization

import java.awt.{BorderLayout, Color}
import javax.swing.JFrame

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import smile.plot._

import scala.collection.JavaConversions._


/**
  * Created by andy on 4/27/16.
  */
class mplot2d[K](drmXY: DrmLike[K], samplePercent: Int = 10)  {
     val drmSize = drmXY.nrow
     val numSamples = (drmSize * (samplePercent/10)).toInt
     val mPlotMatrix: Matrix = drmSampleKRows(drmXY, numSamples, false)
     val arrays: Array[Array[Double]]  = Array.ofDim[Double](mPlotMatrix.numRows(), 2)
     for (i <- 0 until mPlotMatrix.numRows()) {
          arrays(i)(0) = mPlotMatrix(i,0)
          arrays(i)(1) = mPlotMatrix(i,1)
     }

     val canvas: PlotCanvas = ScatterPlot.plot(arrays,Color.BLUE)
     canvas.setTitle("2D Scatter Plot")
     canvas.setAxisLabels("X Axis", "A Long Label")

     val plotPanel :PlotPanel = new PlotPanel(canvas)

     val plotFrame: JFrame = new JFrame("2d Plot")
     plotFrame.setLayout(new BorderLayout())
     plotFrame.add(plotPanel)
     plotFrame.setSize(300,300)
     plotFrame.setVisible(true)
     plotFrame.show()

}
