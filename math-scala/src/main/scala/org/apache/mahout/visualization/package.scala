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
package org.apache.mahout


import org.apache.mahout.math._
import org.apache.mahout.math.drm.{DrmLike, _}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._


package object visualization {

  /**
    * Roll a set of datapoints in a mx3 matrix into a 3D Array()()()
    *
    * @param mxXYZ Matrix of data points x_0 = mx(i,0), x_1 = mx(i,1), x_2 = mx(i,2)
    * @return an Array[Array[Array[Double]]] 3d Array
    */
  def mxXYZ2array3d(mxXYZ: Matrix): Array[Array[Array[Double]]] = {

    // number of datapoints
    val m = mxXYZ.numRows()

    // 3d array to return
    val array3d: Array[Array[Array[Double]]] =  Array.ofDim[Double](m, m, 3)

    // roll a set of 3d points in an m x 3 matrix into a m x m x 3 Array.
    for (i <- 0 until m) {
      for (j <- 0 until m) {
        for (k <- 0 until 3) {
          array3d(i)(j)(k) = mxXYZ(i, k)
        }
      }
    }
    array3d
  }

  /**
    * Syntatic sugar for MSurf class
    * @param drmXYZ
    * @param samplePercent
    * @param setVisible
    * @tparam K
    * @return
    */
  def msurf[K](drmXYZ: DrmLike[K], samplePercent: Double = 1, setVisible: Boolean = true): MahoutPlot =
    new MSurf[K](drmXYZ: DrmLike[K], samplePercent, setVisible)

  /**
    * Syntatic sugar for MPlot2d class
    * @param drmXY
    * @param samplePercent
    * @param setVisible
    * @tparam K
    * @return
    */
  def mpot2d[K](drmXY: DrmLike[K], samplePercent: Double = 1, setVisible: Boolean = true): MahoutPlot =
    new MPlot2d[K](drmXY: DrmLike[K], samplePercent, setVisible)

  /**
    * Syntatic sugar for MPlot3d class
    * @param drmXYZ
    * @param samplePercent
    * @param setVisible
    * @tparam K
    * @return
    */
  def mplot3d[K](drmXYZ: DrmLike[K], samplePercent: Double = 1, setVisible: Boolean = true): MahoutPlot =
    new MPlot3d[K](drmXYZ: DrmLike[K], samplePercent, setVisible)

  /**
    * Syntatic sugar for MGrid class
    * @param drmXYZ
    * @param samplePercent
    * @param setVisible
    * @tparam K
    * @return
    */
  def mgrid[K](drmXYZ: DrmLike[K], samplePercent: Double = 1, setVisible: Boolean = true): MahoutPlot =
    new MGrid[K](drmXYZ: DrmLike[K], samplePercent, setVisible)

  /**
    *
    * @param drmX
    * @param numBins
    * @param samplePercent
    * @param setVisible
    * @tparam K
    * @return
    */
  def mhisto[K](drmX: DrmLike[K], numBins: Int, samplePercent: Double = 1, setVisible: Boolean = true): MahoutPlot =
    new MHisto[K](drmX: DrmLike[K], numBins, samplePercent, setVisible)

  /**
    *
    * @param drmXY
    * @param numBins
    * @param samplePercent
    * @param setVisible
    * @tparam K
    * @return
    */
  def mhisto3d[K](drmXY: DrmLike[K], numBins: Int, samplePercent: Double = 1, setVisible: Boolean = true): MahoutPlot =
    new MHisto3d[K](drmXY: DrmLike[K], numBins, samplePercent, setVisible)


}
