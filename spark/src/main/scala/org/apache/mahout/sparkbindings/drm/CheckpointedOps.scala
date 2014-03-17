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

package org.apache.mahout.sparkbindings.drm

import scala.reflect.ClassTag
import org.apache.mahout.math.{DenseVector, Vector}
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import RLikeDrmOps._
import org.apache.spark.SparkContext._


/**
 * Additional experimental operations over CheckpointedDRM implementation. I will possibly move them up to
 * the DRMBase once they stabilize.
 *
 */
class CheckpointedOps[K: ClassTag](val drm: CheckpointedDrm[K]) {

  /**
   * Reorganize every partition into a single in-core matrix
   * @return
   */
  def blockify(): BlockifiedDrmRdd[K] =
    org.apache.mahout.sparkbindings.drm.blockify(rdd = drm.rdd, blockncol = drm.ncol)

  /** Column sums. At this point this runs on checkpoint and collects in-core vector. */
  def colSums(): Vector = {
    val n = drm.ncol

    drm.rdd
        // Throw away keys
        .map(_._2)
        // Fold() doesn't work with kryo still. So work around it.
        .mapPartitions(iter => {
      val acc = ((new DenseVector(n): Vector) /: iter)((acc, v) => acc += v)
      Iterator(acc)
    })
        // Since we preallocated new accumulator vector per partition, this must not cause any side
        // effects now.
        .reduce(_ += _)

  }

  def colMeans(): Vector = if (drm.nrow == 0) drm.colSums() else drm.colSums() /= drm.nrow

}

