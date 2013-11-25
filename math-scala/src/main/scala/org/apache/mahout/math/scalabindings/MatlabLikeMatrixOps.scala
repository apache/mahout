/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.math.scalabindings

import org.apache.mahout.math.{Vector, Matrix}
import scala.collection.JavaConversions._
import RLikeOps._

class MatlabLikeMatrixOps(_m: Matrix) extends MatrixOps(_m) {

  /**
   * matrix-matrix multiplication
   * @param that
   * @return
   */
  def *(that: Matrix) = m.times(that)

  /**
   * matrix-vector multiplication
   * @param that
   * @return
   */
  def *(that: Vector) = m.times(that)

  /**
   * Hadamard product
   *
   * @param that
   * @return
   */

  private[math] def *@(that: Matrix) = cloned *= that

  private[math] def *@(that: Double) = cloned *= that

  /**
   * in-place Hadamard product. We probably don't want to use assign
   * to optimize for sparse operations, in case of Hadamard product
   * it really can be done
   * @param that
   */
  private[math] def *@=(that: Matrix) = {
    m.zip(that).foreach(t => t._1.vector *= t._2.vector)
    m
  }

  private[math] def *@=(that: Double) = {
    m.foreach(_.vector() *= that)
    m
  }
}

