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

import org.apache.log4j.Logger
import org.apache.mahout.math
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag

package object drm {

  private[drm] final val log = Logger.getLogger("org.apache.mahout.sparkbindings")

  private[sparkbindings] implicit def cpDrm2DrmRddInput[K](cp: CheckpointedDrmSpark[K]): DrmRddInput[K] =
    cp.rddInput

  private[sparkbindings] implicit def cpDrmGeneric2DrmRddInput[K](cp: CheckpointedDrm[K]): DrmRddInput[K] =
    cp.asInstanceOf[CheckpointedDrmSpark[K]]

  private[sparkbindings] implicit def drmRdd2drmRddInput[K:ClassTag](rdd: DrmRdd[K]) = new DrmRddInput[K](Left(rdd))

  private[sparkbindings] implicit def blockifiedRdd2drmRddInput[K:ClassTag](rdd: BlockifiedDrmRdd[K]) = new
      DrmRddInput[K](
    Right(rdd))



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

        // create the block by default as dense.
        // would probably be better to sample a subset of these
        // vectors first before creating the entire matrix.
        // so that we don't have the overhead of creating a full second matrix in
        // the case that the matrix is not dense.
        val block = new DenseMatrix(vectors.length, blockncol)
        var row = 0
        while (row < vectors.length) {
          block(row, ::) := vectors(row)
          row += 1
        }

        // Test the density of the data. If the matrix does not meet the
        // requirements for density, convert the Vectors to a sparse Matrix.
        val resBlock = if (densityAnalysis(block)) {
          block
        } else {
          new SparseRowMatrix(vectors.length, blockncol, vectors, true, false)
        }

        Iterator(keys -> resBlock)
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

        blockKeys.ensuring(blockKeys.length == block.nrow)
        blockKeys.view.zipWithIndex.map {
          case (key, idx) =>
            val v = block(idx, ::) // This is just a view!

            // If a view rather than a concrete vector, clone into a concrete vector in order not to
            // attempt to serialize outer matrix when we save it (Although maybe most often this
            // copying is excessive?)
            // if (v.isInstanceOf[MatrixVectorView]) v = v.cloned
            key -> v
        }
    }
}
