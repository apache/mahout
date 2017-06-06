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

import scala.reflect.ClassTag

/** Logical row-range slicing */
case class OpRowRange(
    override var A: DrmLike[Int],
    val rowRange: Range
    ) extends AbstractUnaryOp[Int, Int] {

  assert(rowRange.head >= 0 && rowRange.last < A.nrow, "row range out of range")

  /**
    * Explicit extraction of key class Tag since traits don't support context bound access; but actual
    * implementation knows it
    */
  override val keyClassTag = ClassTag.Int

  /** R-like syntax for number of rows. */
  def nrow: Long = rowRange.length

  /** R-like syntax for number of columns */
  def ncol: Int = A.ncol

}
