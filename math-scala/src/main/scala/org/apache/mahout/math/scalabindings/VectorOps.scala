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

import org.apache.mahout.math.Vector
import scala.collection.JavaConversions._
import org.apache.mahout.math.function.Functions

/**
 * Syntactic sugar for mahout vectors
 * @param v Mahout vector
 */
class VectorOps(val v: Vector) {

  import VectorOps.v2ops

  def apply(i: Int) = v.get(i)

  def update(i: Int, that: Double) = v.setQuick(i, that)

  /** Warning: we only support consecutive views, step is not supported directly */
  def apply(r: Range) = if (r == ::) v else v.viewPart(r.start, r.length * r.step)

  def update(r: Range, that: Vector) = apply(r) := that

  def sum = v.zSum()

  def :=(that: Vector): Vector = {

    // assign op in Mahout requires same
    // cardinality between vectors .
    // we want to relax it here and require
    // v to have _at least_ as large cardinality
    // as "that".
    if (that.length == v.size())
      v.assign(that)
    else if (that.length < v.size) {
      v.assign(0.0)
      that.nonZeroes().foreach(t => v.setQuick(t.index, t.get))
      v
    } else throw new IllegalArgumentException("Assigner's cardinality less than assignee's")
  }

  def :=(that: Double): Vector = v.assign(that)

  def :=(f: (Int, Double) => Double): Vector = {
    for (i <- 0 until length) v(i) = f(i, v(i))
    v
  }

  def equiv(that: Vector) =
    length == that.length &&
        v.all.view.zip(that.all).forall(t => t._1.get == t._2.get)

  def ===(that: Vector) = equiv(that)

  def !==(that: Vector) = nequiv(that)

  def nequiv(that: Vector) = !equiv(that)

  def unary_- = v.assign(Functions.NEGATE)

  def +=(that: Vector) = v.assign(that, Functions.PLUS)

  def -=(that: Vector) = v.assign(that, Functions.MINUS)

  def +=(that: Double) = v.assign(Functions.PLUS, that)

  def -=(that: Double) = +=(-that)

  def -=:(that: Vector) = v.assign(Functions.NEGATE).assign(that, Functions.PLUS)

  def -=:(that: Double) = v.assign(Functions.NEGATE).assign(Functions.PLUS, that)

  def +(that: Vector) = cloned += that

  def -(that: Vector) = cloned -= that

  def -:(that: Vector) = that.cloned -= v

  def +(that: Double) = cloned += that

  def -(that: Double) = cloned -= that

  def -:(that: Double) = that -=: v.cloned

  def length = v.size()

  def cloned: Vector = v.like := v

  def sqrt = v.cloned.assign(Functions.SQRT)

}

object VectorOps {
  private implicit def v2ops(v: Vector): VectorOps = new VectorOps(v)
}
