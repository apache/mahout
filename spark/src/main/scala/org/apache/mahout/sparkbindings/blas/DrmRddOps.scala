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

package org.apache.mahout.sparkbindings.blas

import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.{SequentialAccessSparseVector, DenseVector}
import org.apache.mahout.sparkbindings.DrmRdd

class DrmRddOps[K](private[blas] val rdd: DrmRdd[K]) {

  /** Turn RDD into dense row-wise vectors if density threshold is exceeded. */
  def densify(threshold: Double = 0.80): DrmRdd[K] = rdd.map({
    case (key, v) =>
      val vd = if (!v.isDense && v.getNumNonZeroElements > threshold * v.length) new DenseVector(v) else v
      key -> vd
  })

  /** Turn rdd into sparse RDD if density threshold is underrun. */
  def sparsify(threshold: Double = 0.80): DrmRdd[K] = rdd.map({
    case (key, v) =>
      val vs = if (v.isDense() && v.getNumNonZeroElements <= threshold * v.length)
        new SequentialAccessSparseVector(v)
      else v
      key -> vs
  })

}
