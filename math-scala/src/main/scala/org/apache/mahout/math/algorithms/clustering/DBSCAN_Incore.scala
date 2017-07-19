/**
  * Licensed to the Apache Software Foundation (ASF) under one
  * or more contributor license agreements. See the NOTICE file
  * distributed with this work for additional information
  * regarding copyright ownership. The ASF licenses this file
  * to you under the Apache License, Version 2.0 (the
  * "License"); you may not use this file except in compliance
  * with the License. You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing,
  * software distributed under the License is distributed on an
  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  * KIND, either express or implied. See the License for the
  * specific language governing permissions and limitations
  * under the License.
  */
package org.apache.mahout.math.algorithms.clustering

import scala.collection.JavaConversions.iterableAsScalaIterable
import org.apache.mahout.math.scalabindings.::
import org.apache.mahout.math.scalabindings.RLikeOps.m2mOps
import org.apache.mahout.math.scalabindings.RLikeOps.v2vOps
import org.apache.mahout.math.scalabindings.dvec
import org.apache.mahout.math._
import scalabindings._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Queue

class DBSCAN_Incore {

  def expandCluster(i: Int, data: DenseMatrix, neighbours: DenseVector, clusterId: Int, Eps: Double, Minpts: Int) = {
    data(i, 3) = clusterId.toDouble
    var neighbourQueue = new Queue[Int]()
    for(index <- 0 until neighbours.length) {
      neighbourQueue += neighbours(index).toInt
    }

    while(neighbourQueue.nonEmpty) {
      var curr: Int = neighbourQueue.dequeue()
      if(data(curr, 1) == 0) {
        var currNeighbours: DenseVector = findNeighbours(curr, data, Eps)
        data(curr, 1) = 1
        if(currNeighbours.length >= Minpts) {
          data(curr, 0) = 1 //coreFlag == True
          for (index <- 0 until currNeighbours.length) {
            neighbourQueue += currNeighbours(index).toInt
          }
        }
        if(data(curr, 3) == -1) {
          data(curr, 3) = clusterId
        }
      }
    }
  }

  def DBSCAN(data: DenseMatrix, Eps: Double, Minpts: Int): Unit = {

    var clusterId: Int = 0
    for(i <- 0 until data.nrow) {
      if(data(i, 1) != 1) {
        //i.e. if notProcessed...
        val neighbours: DenseVector = findNeighbours(i, data, Eps)
        data(i, 1) = 1.0 // ==> processedFlag = True
        if(neighbours.length >= Minpts) {
          // ==> i corresponds to a core point.
          data(i, 0) = 1 // ==> coreFlag = True
          //          data(i, 3) = clusterId
          expandCluster(i, data, neighbours, clusterId, Eps, Minpts)
          clusterId += 1
        }
        else {
          data(i, 4) = 1.0 // ==> noiseFlag = True
        }
      }
    }
  }

  /*
  @args
  point: Int - Id of the point whose neighbours are to be computed
  data: DenseMatrix[Double] - The augmented matrix containing all the data points along with the 4 appended columns
  Returns: A DenseVector containing the ID's of all the points that are at an Eps distance from point.
   */
  def findNeighbours(point: Int, data: DenseMatrix, Eps: Double) : DenseVector = {
    val pointId: Int = data(point, 2).toInt
    var neighbours: ArrayBuffer[Int] = new ArrayBuffer[Int]()
    val pointData: DenseVector = dvec(data(pointId, ::))
    var neighbourCount = 0
    for(row <- data) {
      if(row(0).toInt != pointId) {
        val arg1 = dvec(row)
        val arg2 = dvec(pointData)
        if(distanceMetric(arg1, arg2) <= Eps) {
          neighbourCount += 1
          neighbours += pointData(2).toInt
        }
      }
    }
    var neighboursDvec: DenseVector = dvec(neighbours)
    neighboursDvec
  }

  def addColumns(arg: Array[Double]): Array[Double] = {
    val newArr = new Array[Double](arg.length + 5)
    newArr(0) = 0.0 //coreFlag //Initialize all points as non-core points
    newArr(1) = 0.0 //processedFlag //Initialize all points as non-processed points
    newArr(2) = -1.0 //globalId
    newArr(3) = -1.0 //clusterId //Initialize all points as not belonging to any cluster
    newArr(4) = 0.0 //noiseFlag //Initialize all points as not-Noise

    for (i <- 0 until (arg.size)) {
      newArr(i + 5) = arg(i)
    }
    newArr
  }

  /*
  Takes in two rows as input. Rows that contain the co-ordinates of the data as well as the augmented columns added at the beginning.
  Computes distance of only the data part of the rows, ignoring the augmented columns.
  //DistanceMetric is Euclidean distance as of now, Check and add other types of distance metrics and rename methods accordingly
  Also give user the option of choosing the distance metric while running the algorithm.
   */
  def distanceMetric(arg1: DenseVector, arg2: DenseVector) : Double = {
    var diffsqSum = -1.0
    if(arg1.length != arg2.length) {
      return diffsqSum
    }
    else {
      val diff = arg1 - arg2
      val diffsq = diff^2
      diffsqSum = 0.0
      for(i <- 4 until diffsq.length) {
        diffsqSum += diffsq(i)
      }
      diffsqSum = math.sqrt(diffsqSum)
    }
    diffsqSum
  }

}
