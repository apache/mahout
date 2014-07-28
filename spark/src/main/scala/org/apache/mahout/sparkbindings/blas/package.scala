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

import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings.drm.{CheckpointedDrmSpark, DrmRddInput}
import org.apache.spark.SparkContext._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import scalabindings._
import RLikeOps._

/**
 * This validation contains distributed algorithms that distributed matrix expression optimizer picks
 * from.
 */
package object blas {

  implicit def drmRdd2ops[K:ClassTag](rdd:DrmRdd[K]):DrmRddOps[K] = new DrmRddOps[K](rdd)

  private[mahout] def fixIntConsistency(op: DrmLike[Int], src: DrmRdd[Int]): DrmRdd[Int] = {

    if (op.canHaveMissingRows) {

      val rdd = src
      val sc = rdd.sparkContext
      val dueRows = safeToNonNegInt(op.nrow)
      val dueCols = op.ncol

      // Compute the fix.
      sc

          // Bootstrap full key set
          .parallelize(0 until dueRows, numSlices = rdd.partitions.size max 1)

          // Enable PairedFunctions
          .map(_ -> Unit)

          // Cogroup with all rows
          .cogroup(other = rdd)

          // Filter out out-of-bounds
          .filter { case (key, _) => key >= 0 && key < dueRows}

          // Coalesce and output RHS
          .map { case (key, (seqUnit, seqVec)) =>
        val acc = seqVec.headOption.getOrElse(new SequentialAccessSparseVector(dueCols))
        val vec = if (seqVec.size > 0) (acc /: seqVec.tail)(_ + _) else acc
        key -> vec
      }

    } else src

  }

}
