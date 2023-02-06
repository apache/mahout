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

/**
 * Trait representing a distance metric.
 *
 * @tparam Vector A vector class with properties such as `size` and `aggregate`.
 */
trait DistanceMetric extends Serializable {
    /**
       * Computes the distance between two vectors.
       *
       * @param v1 The first vector.
       * @param v2 The second vector.
       *
       * @return The distance between `v1` and `v2`.
       *
   */
  def distance(v1: Vector, v2: Vector): Double
}

/**
 * Object for selecting a distance metric based on a given name.
 */
object DistanceMetricSelector extends Serializable{

    /**
       * A map of named distance metrics and their associated values.
   */
  val namedMetricLookup = Map('Chebyshev -> 1.0, 'Cosine -> 2.0)

    /**
      * Selects a distance metric based on the provided value.
      *
      * @param dm The value associated with the desired distance metric.
      *
      * @return The distance metric associated with `dm`.
      */
  def select(dm: Double): DistanceMetric = {
    dm match {
      case 1.0 => Chebyshev
      case 2.0 => Cosine
    }
  }
}

/**
 * Object representing the Chebyshev distance metric.
 */
object Chebyshev extends DistanceMetric {
/**
   * Computes the Chebyshev distance between two vectors.
   *
   * @param v1 The first vector.
   * @param v2 The second vector.
   *
   * @return The Chebyshev distance between `v1` and `v2`.
   *
   * @throws CardinalityException If `v1` and `v2` have different dimensions.
   */
  def distance(v1: Vector, v2: Vector): Double =  {
    if (v1.size != v2.size) throw new CardinalityException(v1.size, v2.size)
    v1.aggregate(v2, Functions.MAX_ABS, Functions.MINUS)
  }
}

/**
 * Object representing the Cosine distance metric.
 */
object Cosine extends DistanceMetric {
    /**
       * Computes the Cosine distance between two vectors.
       *
       * @param v1 The first vector.
       * @param v2 The second vector.
       *
       * @return The Cosine distance between `v1` and `v2`.
       */
  def distance(v1: Vector, v2: Vector): Double = 1.0 - v1.dot(v2) / (Math.sqrt(v1.getLengthSquared) * Math.sqrt(v2.getLengthSquared))
}
