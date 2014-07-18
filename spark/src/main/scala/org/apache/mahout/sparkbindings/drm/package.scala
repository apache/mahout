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

package org.apache.mahout.sparkbindings

import org.apache.mahout.math._
import org.apache.spark.SparkContext
import scala.collection.JavaConversions._
import org.apache.hadoop.io.{LongWritable, Text, IntWritable, Writable}
import org.apache.log4j.Logger
import java.lang.Math
import org.apache.spark.rdd.{FilteredRDD, RDD}
import scala.reflect.ClassTag
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.spark.broadcast.Broadcast
import org.apache.mahout.math.drm._
import org.apache.mahout.sparkbindings.drm.classification.NaiveBayes
import SparkContext._
import org.apache.mahout.math


package object drm {

  private[drm] final val log = Logger.getLogger("org.apache.mahout.sparkbindings");

  private[sparkbindings] implicit def input2drmRdd[K](input: DrmRddInput[K]): DrmRdd[K] = input.toDrmRdd()

  private[sparkbindings] implicit def input2blockifiedDrmRdd[K](input: DrmRddInput[K]): BlockifiedDrmRdd[K] = input.toBlockifiedDrmRdd()

  private[sparkbindings] implicit def cpDrm2DrmRddInput[K: ClassTag](cp: CheckpointedDrm[K]): DrmRddInput[K] =
    new DrmRddInput(rowWiseSrc = Some(cp.ncol -> cp.rdd))

//  /** Broadcast vector (Mahout vectors are not closure-friendly, use this instead. */
//  private[sparkbindings] def drmBroadcast(x: Vector)(implicit sc: SparkContext): Broadcast[Vector] = sc.broadcast(x)
//
//  /** Broadcast in-core Mahout matrix. Use this instead of closure. */
//  private[sparkbindings] def drmBroadcast(m: Matrix)(implicit sc: SparkContext): Broadcast[Matrix] = sc.broadcast(m)

  /** Implicit broadcast cast for Spark physical op implementations. */
  private[sparkbindings] implicit def bcast2val[K](bcast:Broadcast[K]):K = bcast.value

  private[sparkbindings] def blockify[K: ClassTag](rdd: DrmRdd[K], blockncol: Int): BlockifiedDrmRdd[K] = {

    rdd.mapPartitions(iter => {

      if (iter.isEmpty) {
        Iterator.empty
      } else {

        val data = iter.toIterable
        val keys = data.map(t => t._1).toArray[K]
        val vectors = data.map(t => t._2).toArray

        val block = if (vectors(0).isDense) {
          val block = new DenseMatrix(vectors.size, blockncol)
          var row = 0
          while (row < vectors.size) {
            block(row, ::) := vectors(row)
            row += 1
          }
          block
        } else {
          new SparseRowMatrix(vectors.size, blockncol, vectors)
        }

        Iterator(keys -> block)
      }
    })
  }

  /** Performs rbind() on all blocks inside same partition to ensure there's only one block here. */
  private[sparkbindings] def rbind[K: ClassTag](rdd: BlockifiedDrmRdd[K]): BlockifiedDrmRdd[K] =
    rdd.mapPartitions(iter => {
      if (iter.isEmpty) {
        Iterator.empty
      } else {
        Iterator(math.drm.rbind(iter.toIterable))
      }
    })

  private[sparkbindings] def deblockify[K: ClassTag](rdd: BlockifiedDrmRdd[K]): DrmRdd[K] =

  // Just flat-map rows, connect with the keys
    rdd.flatMap {
      case (blockKeys: Array[K], block: Matrix) =>

        blockKeys.ensuring(blockKeys.size == block.nrow)
        blockKeys.view.zipWithIndex.map {
          case (key, idx) =>
            var v = block(idx, ::) // This is just a view!

            // If a view rather than a concrete vector, clone into a concrete vector in order not to
            // attempt to serialize outer matrix when we save it (Although maybe most often this
            // copying is excessive?)
            // if (v.isInstanceOf[MatrixVectorView]) v = v.cloned
            key -> v
        }
    }

  /**
   * Distributed training of a Naive Bayes model. Follows the approach presented in Rennie et.al.: Tackling the poor
   * assumptions of Naive Bayes Text classifiers, ICML 2003, http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
   *
   * @param observationsPerLabel an array of matrices. Every matrix contains the observations for a particular label.
   * @param trainComplementary whether to train a complementary Naive Bayes model
   * @param alphaI smoothing parameter
   * @return trained naive bayes model
   */
  def trainNB[K: ClassTag](observationsPerLabel: Array[DrmLike[K]],
      alphaI: Float = NaiveBayes.defaultAlphaI) = NaiveBayes.trainNB(observationsPerLabel, alphaI)

}
