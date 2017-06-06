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

import org.apache.mahout.math.Vector
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._

import scala.reflect.ClassTag

/** Logical A'x. */
case class OpAtx(
    override var A: DrmLike[Int],
    val x: Vector
    ) extends AbstractUnaryOp[Int, Int] {

  override protected[mahout] lazy val partitioningTag: Long = A.partitioningTag

  assert(A.nrow == x.length, "Incompatible operand geometry")

  /**
    * Explicit extraction of key class Tag since traits don't support context bound access; but actual
    * implementation knows it
    */
  override val keyClassTag = ClassTag.Int

  /** R-like syntax for number of rows. */
  def nrow: Long = safeToNonNegInt(A.ncol)

  /** R-like syntax for number of columns */
  def ncol: Int = 1

}
