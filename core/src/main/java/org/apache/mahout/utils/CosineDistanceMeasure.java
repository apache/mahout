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

import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.matrix.CardinalityException;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.parameters.Parameter;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;

/**
 * This class implements a cosine distance metric by dividing the dot product of two vectors by the product of their
 * lengths
 */
public class CosineDistanceMeasure implements DistanceMeasure {

  @Override
  public void configure(JobConf job) {
    // nothing to do
  }

  @Override
  public Collection<Parameter<?>> getParameters() {
    return Collections.emptyList();
  }

  @Override
  public void createParameters(String prefix, JobConf jobConf) {
    // nothing to do
  }

  public static double distance(double[] p1, double[] p2) {
    double dotProduct = 0.0;
    double lengthSquaredp1 = 0.0;
    double lengthSquaredp2 = 0.0;
    for (int i = 0; i < p1.length; i++) {
      lengthSquaredp1 += p1[i] * p1[i];
      lengthSquaredp2 += p2[i] * p2[i];
      dotProduct += p1[i] * p2[i];
    }
    double denominator = Math.sqrt(lengthSquaredp1) * Math.sqrt(lengthSquaredp2);

    // correct for floating-point rounding errors
    if (denominator < dotProduct) {
      denominator = dotProduct;
    }

    return 1.0 - (dotProduct / denominator);
  }

  @Override
  public double distance(Vector v1, Vector v2) {
    if (v1.size() != v2.size()) {
      throw new CardinalityException();
    }
    double lengthSquaredv1 = 0.0;
    Iterator<Vector.Element> iter = v1.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element elt = iter.next();
      lengthSquaredv1 += elt.get() * elt.get();
    }
    iter = v2.iterateNonZero();
    double lengthSquaredv2 = 0.0;
    while (iter.hasNext()) {
      Vector.Element elt = iter.next();
      lengthSquaredv2 += elt.get() * elt.get();
    }

    double dotProduct = v1.dot(v2);
    double denominator = Math.sqrt(lengthSquaredv1) * Math.sqrt(lengthSquaredv2);

    // correct for floating-point rounding errors
    if (denominator < dotProduct) {
      denominator = dotProduct;
    }

    return 1.0 - (dotProduct / denominator);
  }

  @Override
  public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
    return distance(centroid, v); // TODO
  }

}
