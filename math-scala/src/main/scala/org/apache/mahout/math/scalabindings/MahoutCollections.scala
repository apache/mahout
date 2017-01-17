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

class MahoutVectorInterfaces(v: Vector) {
  /** Convert to Array[Double] */
  def toArray: Array[Double] = {
    var a = new Array[Double](v.size)
    for (i <- 0 until v.size){
      a(i) = v.get(i)
    }
    a
  }

  /** Convert to Map[Int, Double] */
  def toMap: Map[Int, Double] = {
    import collection.JavaConverters._
    val ms = collection.mutable.Map[Int, Double]()
    for (e <- v.nonZeroes().asScala) {
      ms += (e.index -> e.get)
    }
    ms.toMap
  }

}

object MahoutCollections {
  implicit def v2scalaish(v: Vector) = new MahoutVectorInterfaces(v)
}
