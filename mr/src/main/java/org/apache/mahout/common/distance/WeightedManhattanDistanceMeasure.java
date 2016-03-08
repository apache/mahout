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

package org.apache.mahout.common.distance;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

/**
 * This class implements a "Manhattan distance" metric by summing the absolute values of the difference
 * between each coordinate, optionally with weights.
 */
public class WeightedManhattanDistanceMeasure extends WeightedDistanceMeasure {
  
  @Override
  public double distance(Vector p1, Vector p2) {
    double result = 0;
    
    Vector res = p2.minus(p1);
    if (getWeights() == null) {
      for (Element elt : res.nonZeroes()) {
        result += Math.abs(elt.get());
      }
      
    } else {
      for (Element elt : res.nonZeroes()) {
        result += Math.abs(elt.get() * getWeights().get(elt.index()));
      }
    }
    
    return result;
  }
  
  @Override
  public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
    return distance(centroid, v); // TODO
  }
  
}
