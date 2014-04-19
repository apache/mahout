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

/**
 * This class implements a cosine distance metric by dividing the dot product of two vectors by the product of their
 * lengths.  That gives the cosine of the angle between the two vectors.  To convert this to a usable distance,
 * 1-cos(angle) is what is actually returned.
 */
public class CosineDistanceMeasure implements DistanceMeasure {
  
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
    
    // correct for zero-vector corner case
    if (denominator == 0 && dotProduct == 0) {
      return 0;
    }
    
    return 1.0 - dotProduct / denominator;
  }
  
  @Override
  public double distance(Vector v1, Vector v2) {
    if (v1.size() != v2.size()) {
      throw new CardinalityException(v1.size(), v2.size());
    }
    double lengthSquaredv1 = v1.getLengthSquared();
    double lengthSquaredv2 = v2.getLengthSquared();
    
    double dotProduct = v2.dot(v1);
    double denominator = Math.sqrt(lengthSquaredv1) * Math.sqrt(lengthSquaredv2);
    
    // correct for floating-point rounding errors
    if (denominator < dotProduct) {
      denominator = dotProduct;
    }
    
    // correct for zero-vector corner case
    if (denominator == 0 && dotProduct == 0) {
      return 0;
    }
    
    return 1.0 - dotProduct / denominator;
  }
  
  @Override
  public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
    
    double lengthSquaredv = v.getLengthSquared();
    
    double dotProduct = v.dot(centroid);
    double denominator = Math.sqrt(centroidLengthSquare) * Math.sqrt(lengthSquaredv);
    
    // correct for floating-point rounding errors
    if (denominator < dotProduct) {
      denominator = dotProduct;
    }
    
    // correct for zero-vector corner case
    if (denominator == 0 && dotProduct == 0) {
      return 0;
    }
    
    return 1.0 - dotProduct / denominator;
  }
  
}
