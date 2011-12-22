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

package org.apache.mahout.clustering.dirichlet.models;

import java.util.Iterator;

import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.clustering.dirichlet.UncommonDistributions;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

public class GaussianCluster extends AbstractCluster {
  
  public GaussianCluster() {}
  
  public GaussianCluster(Vector point, int id2) {
    super(point, id2);
  }
  
  public GaussianCluster(Vector center, Vector radius, int id) {
    super(center, radius, id);
  }
  
  @Override
  public String getIdentifier() {
    return "GC:" + getId();
  }
  
  @Override
  public Model<VectorWritable> sampleFromPosterior() {
    return new GaussianCluster(getCenter(), getRadius(), getId());
  }
  
  @Override
  public double pdf(VectorWritable vw) {
    Vector x = vw.get();
    Vector m = getCenter();
    Vector s = getRadius().plus(0.0000001); // add a small prior to avoid divide by zero
    return Math.exp(-(divideSquareAndSum(x.minus(m), s) / 2)) / zProdSqt2Pi(s);
  }
  
  private double zProdSqt2Pi(Vector s) {
    double prod = 1;
    for (int i = 0; i < s.size(); i++) {
      prod *= s.getQuick(i) * UncommonDistributions.SQRT2PI;
    }
    return prod;
  }
  
  private double divideSquareAndSum(Vector numerator, Vector denominator) {
    double result = 0;
    for (Iterator<Element> it = denominator.iterateNonZero(); it.hasNext();) {
      Element denom = it.next();
      double quotient = numerator.getQuick(denom.index()) / denom.get();
      result += quotient * quotient;
    }
    return result;
  }
  
}
