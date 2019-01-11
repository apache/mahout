/**
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
package org.apache.mahout.math.algorithms.common.distance

import org.apache.mahout.math.function.Functions
import org.apache.mahout.math.{CardinalityException, Vector}

trait DistanceMetric extends Serializable {
  def distance(v1: Vector, v2: Vector): Double
}


object DistanceMetricSelector extends Serializable{

  val namedMetricLookup = Map('Chebyshev -> 1.0, 'Cosine -> 2.0)

  def select(dm: Double): DistanceMetric = {
    dm match {
      case 1.0 => Chebyshev
      case 2.0 => Cosine
    }
  }
}

object Chebyshev extends DistanceMetric {
  def distance(v1: Vector, v2: Vector): Double =  {
    if (v1.size != v2.size) throw new CardinalityException(v1.size, v2.size)
    v1.aggregate(v2, Functions.MAX_ABS, Functions.MINUS)
  }
}

object Cosine extends DistanceMetric {
  def distance(v1: Vector, v2: Vector): Double = 1.0 - v1.dot(v2) / (Math.sqrt(v1.getLengthSquared) * Math.sqrt(v2.getLengthSquared))
}
