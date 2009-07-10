/**
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

package org.apache.mahout.utils;

import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.parameters.Parametered;

/** This interface is used for objects which can determine a distance metric between two points */
public interface DistanceMeasure extends Parametered {

  /**
   * Returns the distance metric applied to the arguments
   *
   * @param v1 a Vector defining a multidimensional point in some feature space
   * @param v2 a Vector defining a multidimensional point in some feature space
   * @return a scalar doubles of the distance
   */
  double distance(Vector v1, Vector v2);

  /**
   * Optimized version of distance metric for sparse vectors. This distance computation requires operations proportional
   * to the number of non-zero elements in the vector instead of the cardinality of the vector.
   *
   * @param centroidLengthSquare Square of the length of centroid
   * @param centroid             Centroid vector
   */
  double distance(double centroidLengthSquare, Vector centroid, Vector v);

}
