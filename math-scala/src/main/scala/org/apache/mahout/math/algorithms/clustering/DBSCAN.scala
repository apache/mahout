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
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math._
import org.apache.mahout.math.algorithms.common.distance.{DistanceMetric, DistanceMetricSelector}
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._

import scala.collection.mutable
import scala.io.Source

class DistributedDBSCAN extends ClusteringFitter {

  var epsilon: Double = _
  var minPts: Int = _
  var distanceMeasure: Symbol = _

  def setStandardHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    epsilon = hyperparameters.asInstanceOf[Map[Symbol, Double]].getOrElse('epsilon, 0.5)
    minPts = hyperparameters.asInstanceOf[Map[Symbol, Int]].getOrElse('minPts, 1)
    distanceMeasure = hyperparameters.asInstanceOf[Map[Symbol, Symbol]].getOrElse('distanceMeasure, 'Cosine)
  }

  def fit[K](input: DrmLike[K],
             hyperparameters: (Symbol, Any)*): DBSCANModel = {

    setStandardHyperparameters(hyperparameters.toMap)
    implicit val ctx = input.context
    implicit val ktag =  input.keyClassTag

    val dmNumber = DistanceMetricSelector.namedMetricLookup(distanceMeasure)

    val configBC = drmBroadcast(dvec(epsilon, minPts, dmNumber))

    val clusters = input.allreduceBlock(
      {
        // Assign All Points to Clusters
        case (keys, block: Matrix) => {
          val epsilon_local = configBC.value.get(0)
          val minPts_local = configBC.value.get(1)

          val distanceMetric = DistanceMetricSelector.select(configBC.value.get(3))
          val icDBSCAN = new InCoreDBSCAN(block, epsilon_local, minPts_local.toInt, distanceMetric)
          // do stuff on icDBSCAN
          icDBSCAN.data
        }
      }, {
        // Optionally Merge Clusters that are close enough
        case (oldM: Matrix, newM: Matrix) => {
          // this does nothing- just returns the left matrix
          oldM
        }
      })

    val model = new DBSCANModel(1)
    model.summary = s"""foo the bar"""
    model
  }

}

class DBSCANModel(args: Int) extends ClusteringModel {
  def cluster[K](input: DrmLike[K]): DrmLike[K] ={
    input
  }
}

object Test{
  def readInputAndMakeMatrix(filename: String): Matrix = {
    val iter = Source.fromFile(filename)
    val lines = iter.getLines()
    val rows = lines.next().toInt
    val cols = lines.next().toInt
    var input: Matrix = new DenseMatrix(rows, cols)
    var count = 0
    while(lines.hasNext) {
      val str = lines.next().split(' ')
      for(i <- 0 until str.length) {
        input(count, i) = str(i).toDouble
      }
      count = count + 1
    }
    input
  }
}

class InCoreDBSCAN(input: Matrix, epsilon: Double, minPts: Int) extends Serializable {

  var data: Matrix = input
  val eps: Double = epsilon
  val minpts: Int = minPts

  var coreCount = 0
  var noiseCount = 0
  var clusterCount = 0

  var metadata: Matrix = new DenseMatrix(input.numRows(), 6) //0-Id, 1-Processed, 2-Core, 3-ClusterId, 4-ParentId, 5-Noise

  def expandCluster(i: Int, neighbours: DenseVector, clusterId: Int): Unit = {

    metadata(i, 3) = clusterId.toDouble
    var neighbourQueue = new mutable.Queue[Int]()
    for(index <- 0 until neighbours.length) {
      neighbourQueue += neighbours(index).toInt
    }

    while(neighbourQueue.nonEmpty) {
      var curr: Int = neighbourQueue.dequeue()
      if(metadata(curr, 1) != 1) {
        var currNeighbours: DenseVector = findNeighbours(curr)
        metadata(curr, 1) = 1
        if(currNeighbours.length >= minpts) {
          metadata(curr, 2) = 1 //coreFlag == True
          coreCount = coreCount + 1
          for (index <- 0 until currNeighbours.length) {
            neighbourQueue += currNeighbours(index).toInt
          }
        }
        if(metadata(curr, 3) == -1) {
          metadata(curr, 3) = clusterId
          if(metadata(curr, 5) == 1)metadata(curr, 5) = -1
        }
      }
    }
  }

  def interpretResults(metadata: Matrix): Unit = {
    for(i <- 0 until metadata.numRows()) {
      if(metadata(i, 5) == 1)noiseCount = noiseCount + 1
    }
    println("CoreCount = " + coreCount)
    println("ClusterCount = " + clusterCount)
    println("NoiseCount = " + noiseCount)
  }

  def DBSCAN(): Matrix = {
    coreCount = 0
    clusterCount = 0
    noiseCount = 0
    for(i <- 0 until input.numRows()) {
      metadata(i,0) = i
      metadata(i,1) = -1
      metadata(i,2) = -1
      metadata(i,3) = -1
      metadata(i,4) = i
      metadata(i,5) = -1
    }

    var clusterId: Int = 0
    for(point <- 0 until input.numRows()) {

      if(metadata(point, 1) != 1) { //if point is not processed.
        val neighbours: DenseVector = findNeighbours(point)
        metadata(point, 1) = 1 // set processedFlag = True
        if(neighbours.length >= minpts) { //corePoint
          metadata(point, 2) = 1 // ==> coreFlag = True
          coreCount = coreCount + 1
          expandCluster(point, neighbours, clusterId)
          clusterId += 1
        }
        else {
          metadata(point, 5) = 1 // set noiseFlag = True
        }
      }
    }
    clusterCount = clusterId
    interpretResults(metadata)
    metadata
  }

  /*
  @args
  point: Int - Id of the point whose neighbours are to be computed
  data: DenseMatrix[Double] - The augmented matrix containing all the data points along with the 4 appended columns
  Returns: A DenseVector containing the ID's of all the points that are at an Eps distance from point.
   */
  def findNeighbours(point: Int) : DenseVector = {
    val pointId: Int = point
    var neighbours: mutable.ArrayBuffer[Int] = new mutable.ArrayBuffer[Int]()
    var neighbourCount = 0
    for(row <- 0 until data.numRows()) {
      if(row != pointId) {
        val arg1 = dvec(data(row, ::))
        val arg2 = dvec(data(pointId, ::))
        if(distanceMetric(arg1, arg2) <= eps) {
          neighbourCount += 1
          neighbours += row.toInt
        }
      }
    }
    var neighboursDvec: DenseVector = dvec(neighbours)
    neighboursDvec
  }
  //
  //  def addColumns(arg: Array[Double]): Array[Double] = {
  //    val newArr = new Array[Double](arg.length + 5)
  //    newArr(0) = 0.0 //coreFlag //Initialize all points as non-core points
  //    newArr(1) = 0.0 //processedFlag //Initialize all points as non-processed points
  //    newArr(2) = -1.0 //globalId
  //    newArr(3) = -1.0 //clusterId //Initialize all points as not belonging to any cluster
  //    newArr(4) = 0.0 //noiseFlag //Initialize all points as not-Noise
  //
  //    for (i <- 0 until (arg.size)) {
  //      newArr(i + 5) = arg(i)
  //    }
  //    newArr
  //  }

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
      for(i <- 0 until diffsq.length) {
        diffsqSum += diffsq(i)
      }
      diffsqSum = Math.sqrt(diffsqSum)
    }
    diffsqSum
  }

}