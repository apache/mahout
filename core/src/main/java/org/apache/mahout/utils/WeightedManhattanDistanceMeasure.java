package org.apache.mahout.utils;

/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.matrix.Vector;

/**
 * This class implements a "manhattan distance" metric by summing the absolute
 * values of the difference between each coordinate, optionally with weights.
 */
public class WeightedManhattanDistanceMeasure extends WeightedDistanceMeasure {

  /* (non-Javadoc)
   * @see org.apache.mahout.utils.DistanceMeasure#distance(org.apache.mahout.matrix.Vector,
   * org.apache.mahout.matrix.Vector)
   */
  public double distance(Vector p1, Vector p2) {
    double result = 0;

    Vector res = p2.minus(p1);
    if (weights == null) {
      for (int i = 0; i < res.cardinality(); i++) {
        result += Math.abs(res.get(i));
      }
    }
    else {
      for (int i = 0; i < res.cardinality(); i++) {
        result += Math.abs(res.get(i) * weights.get(i)); // todo this is where the weights goes, right?
      }
    }

    return result;
  }

}
