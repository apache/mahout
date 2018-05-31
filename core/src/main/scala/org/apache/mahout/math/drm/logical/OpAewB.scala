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

package org.apache.mahout.math.drm.logical

import scala.reflect.ClassTag
import org.apache.mahout.math.drm.DrmLike
import scala.util.Random

/** DRM elementwise operator */
case class OpAewB[K](
    override var A: DrmLike[K],
    override var B: DrmLike[K],
    val op: String
    ) extends AbstractBinaryOp[K, K, K] {


  assert(A.ncol == B.ncol, "arguments must have same number of columns")
  assert(A.nrow == B.nrow, "arguments must have same number of rows")
  assert(A.keyClassTag == B.keyClassTag, "Arguments of elementwise operators must have the same row key")

  override protected[mahout] lazy val partitioningTag: Long =
    if (A.partitioningTag == B.partitioningTag) A.partitioningTag
    else Random.nextLong()

  /**
    * Explicit extraction of key class Tag since traits don't support context bound access; but actual
    * implementation knows it
    */
  override def keyClassTag: ClassTag[K] = A.keyClassTag

  /** R-like syntax for number of rows. */
  def nrow: Long = A.nrow

  /** R-like syntax for number of columns */
  def ncol: Int = A.ncol

}
