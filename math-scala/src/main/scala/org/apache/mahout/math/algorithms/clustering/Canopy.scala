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



import org.apache.mahout.math.algorithms.common.distance.{DistanceMetric, DistanceMetricSelector}
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.function.VectorFunction
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.{Matrix, Vector}


class CanopyClusteringModel(canopies: Matrix, dm: Symbol) extends ClusteringModel {

  val canopyCenters = canopies
  val distanceMetric = dm

  def cluster[K](input: DrmLike[K]): DrmLike[K] = {

    implicit val ctx = input.context
    implicit val ktag =  input.keyClassTag

    val bcCanopies = drmBroadcast(canopyCenters)
    val bcDM = drmBroadcast(dvec(DistanceMetricSelector.namedMetricLookup(distanceMetric)))

    input.mapBlock(1) {
      case (keys, block: Matrix) => {
        val outputMatrix = new DenseMatrix(block.nrow, 1)

        val localCanopies: Matrix = bcCanopies.value
        for (i <- 0 until block.nrow) {
          val distanceMetric = DistanceMetricSelector.select(bcDM.value.get(0))

          val cluster = (0 until localCanopies.nrow).foldLeft(-1, 9999999999999999.9)((l, r) => {
            val dist = distanceMetric.distance(localCanopies(r, ::), block(i, ::))
            if ((dist) < l._2) {
              (r, dist)
            }
            else {
              l
            }
          })._1
          outputMatrix(i, ::) = dvec(cluster)
        }
        keys -> outputMatrix
      }
    }
  }
}


class CanopyClustering extends ClusteringFitter {

  var t1: Double = _  // loose distance
  var t2: Double = _  // tight distance
  var t3: Double = _
  var t4: Double = _
  var distanceMeasure: Symbol = _

  def setStandardHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    t1 = hyperparameters.asInstanceOf[Map[Symbol, Double]].getOrElse('t1, 0.5)
    t2 = hyperparameters.asInstanceOf[Map[Symbol, Double]].getOrElse('t2, 0.1)
    t3 = hyperparameters.asInstanceOf[Map[Symbol, Double]].getOrElse('t3, t1)
    t4 = hyperparameters.asInstanceOf[Map[Symbol, Double]].getOrElse('t4, t2)

    distanceMeasure = hyperparameters.asInstanceOf[Map[Symbol, Symbol]].getOrElse('distanceMeasure, 'Cosine)

  }

  def fit[K](input: DrmLike[K],
             hyperparameters: (Symbol, Any)*): CanopyClusteringModel = {

    setStandardHyperparameters(hyperparameters.toMap)
    implicit val ctx = input.context
    implicit val ktag =  input.keyClassTag

    val dmNumber = DistanceMetricSelector.namedMetricLookup(distanceMeasure)

    val distanceBC = drmBroadcast(dvec(t1,t2,t3,t4, dmNumber))
    val canopies = input.allreduceBlock(
      {

        // Assign All Points to Clusters
        case (keys, block: Matrix) => {
          val t1_local = distanceBC.value.get(0)
          val t2_local = distanceBC.value.get(1)
          val dm = distanceBC.value.get(4)
          CanopyFn.findCenters(block, DistanceMetricSelector.select(dm), t1_local, t2_local)
        }
      }, {
        // Optionally Merge Clusters that are close enough
        case (oldM: Matrix, newM: Matrix) => {
          val t3_local = distanceBC.value.get(2)
          val t4_local = distanceBC.value.get(3)
          val dm = distanceBC.value.get(4)
          CanopyFn.findCenters(oldM, DistanceMetricSelector.select(dm), t3_local, t4_local)
        }
      })

    val model = new CanopyClusteringModel(canopies, distanceMeasure)
    model.summary = s"""CanopyClusteringModel\n${canopies.nrow} Clusters\n${distanceMeasure} distance metric used for calculating distances\nCanopy centers stored in model.canopies where row n coresponds to canopy n"""
    model
  }


}

object CanopyFn extends Serializable {
  def findCenters(block: Matrix, distanceMeasure: DistanceMetric, t1: Double, t2: Double): Matrix = {
    var rowAssignedToCanopy = Array.fill(block.nrow) { false }
    val clusterBuf = scala.collection.mutable.ListBuffer.empty[org.apache.mahout.math.Vector]
    while (rowAssignedToCanopy.contains(false)) {
      val rowIndexOfNextUncanopiedVector = rowAssignedToCanopy.indexOf(false)
      clusterBuf += block(rowIndexOfNextUncanopiedVector, ::).cloned
      block(rowIndexOfNextUncanopiedVector, ::) = svec(Nil, cardinality = block.ncol)
      rowAssignedToCanopy(rowIndexOfNextUncanopiedVector) = true
      for (i <- 0 until block.nrow) {
        if (block(i, ::).getNumNonZeroElements > 0) { //
          distanceMeasure.distance(block(i, ::), clusterBuf.last) match {
            case d if d < t2 => {

              rowAssignedToCanopy(i) = true
              block(i, ::) = svec(Nil, cardinality = block.ncol)
            }
            case d if d < t1 => {

              rowAssignedToCanopy(i) = true
            }
            case d => {}
          }
        }
      }
    }
    dense(clusterBuf)
  }
}