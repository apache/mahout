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

/** Logical A'B */
case class OpAtB[A](
    override var A: DrmLike[A],
    override var B: DrmLike[A])
    extends AbstractBinaryOp[A, A, Int] {

  assert(A.nrow == B.nrow, "Incompatible operand geometry")

  /**
    * Explicit extraction of key class Tag since traits don't support context bound access; but actual
    * implementation knows it
    */
  override def keyClassTag = ClassTag.Int

  /** R-like syntax for number of rows. */
  def nrow: Long = A.ncol

  /** R-like syntax for number of columns */
  def ncol: Int = B.ncol

  /** Non-zero element count */
  def nNonZero: Long =
  // TODO: for purposes of cost calculation, approximate based on operands
    throw new UnsupportedOperationException

}
