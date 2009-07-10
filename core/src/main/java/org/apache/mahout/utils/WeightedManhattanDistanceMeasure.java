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

import java.util.Iterator;

/**
 * This class implements a "manhattan distance" metric by summing the absolute values of the difference between each
 * coordinate, optionally with weights.
 */
public class WeightedManhattanDistanceMeasure extends WeightedDistanceMeasure {

  @Override
  public double distance(Vector p1, Vector p2) {
    double result = 0;

    Vector res = p2.minus(p1);
    if (weights == null) {
      Iterator<Vector.Element> iter = res.iterateNonZero();
      while (iter.hasNext()) {
        Vector.Element elt = iter.next();
        result += Math.abs(elt.get());
      }

    } else {
      Iterator<Vector.Element> iter = res.iterateNonZero();
      while (iter.hasNext()) {
        Vector.Element elt = iter.next();
        result += Math.abs(elt.get() * weights.get(elt.index()));
      }
    }

    return result;
  }

  @Override
  public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
    return distance(centroid, v); // TODO
  }

}
