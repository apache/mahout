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

package org.apache.mahout.math.scalabindings

import org.apache.mahout.math._

class RLikeDoubleScalarOps(val x:Double) extends AnyVal{

  import RLikeOps._

  def +(that:Matrix) = that + x

  def +(that:Vector) = that + x

  def *(that:Matrix) = that * x

  def *(that:Vector) = that * x

  def -(that:Matrix) = x -: that

  def -(that:Vector) = x -: that

  def /(that:Matrix) = x /: that

  def /(that:Vector) = x /: that
  
  def cbind(that:Matrix) = {
    val mx = that.like(that.nrow, that.ncol + 1)
    mx(::, 1 until mx.ncol) := that
    if (x != 0.0) mx(::, 0) := x
    mx
  }

  def rbind(that: Matrix) = {
    val mx = that.like(that.nrow + 1, that.ncol)
    mx(1 until mx.nrow, ::) := that
    if (x != 0.0) mx(0, ::) := x
    mx
  }

  def c(that: Vector): Vector = {
    val cv = that.like(that.length + 1)
    cv(1 until cv.length) := that
    cv(0) = x
    cv
  }

}
