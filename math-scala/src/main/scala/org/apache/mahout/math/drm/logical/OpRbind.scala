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

import org.apache.mahout.math.drm.DrmLike
import scala.util.Random

/** rbind() logical operator */
case class OpRbind[K](
    override var A: DrmLike[K],
    override var B: DrmLike[K]
    ) extends AbstractBinaryOp[K, K, K] {

  assert(A.ncol == B.ncol, "arguments must have same number of columns")
  require(A.keyClassTag == B.keyClassTag, "arguments of rbind() must have the same row key type")

  override protected[mahout] lazy val partitioningTag: Long = Random.nextLong()

  /**
    * Explicit extraction of key class Tag since traits don't support context bound access; but actual
    * implementation knows it
    */
  override def keyClassTag = A.keyClassTag

  /** R-like syntax for number of rows. */
  def nrow: Long = A.nrow + B.nrow

  /** R-like syntax for number of columns */
  def ncol: Int = A.ncol

}
