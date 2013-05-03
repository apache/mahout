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

import java.util.Collection;
import java.util.Collections;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.parameters.Parameter;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

/**
 * This class implements a "manhattan distance" metric by summing the absolute values of the difference
 * between each coordinate
 */
public class ManhattanDistanceMeasure implements DistanceMeasure {

  public static double distance(double[] p1, double[] p2) {
    double result = 0.0;
    for (int i = 0; i < p1.length; i++) {
      result += Math.abs(p2[i] - p1[i]);
    }
    return result;
  }

  @Override
  public void configure(Configuration job) {
  // nothing to do
  }

  @Override
  public Collection<Parameter<?>> getParameters() {
    return Collections.emptyList();
  }

  @Override
  public void createParameters(String prefix, Configuration jobConf) {
  // nothing to do
  }

  @Override
  public double distance(Vector v1, Vector v2) {
    if (v1.size() != v2.size()) {
      throw new CardinalityException(v1.size(), v2.size());
    }
    return v1.aggregate(v2, Functions.PLUS, Functions.MINUS_ABS);
  }

  @Override
  public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
    return distance(centroid, v); // TODO
  }
}
