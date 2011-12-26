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
  
  /* (non-Javadoc)
   * @see org.apache.mahout.clustering.AbstractCluster#setRadius(org.apache.mahout.math.Vector)
   */
  @Override
  protected void setRadius(Vector s2) {
    super.setRadius(s2);
    computeProd2piR();
  }

  // the value of the zProduct(S*2pi) term. Calculated below.
  private double zProd2piR;
  
  /**
   * Compute the product(r[i]*SQRT2PI) over all i. Note that the cluster Radius
   * corresponds to the Stdev of a Gaussian and the Center to its Mean.
   */
  private void computeProd2piR() {
    zProd2piR = 1.0;
    for (Iterator<Element> it = getRadius().iterateNonZero(); it.hasNext();) {
      Element radius = it.next();
      zProd2piR *= radius.get() * UncommonDistributions.SQRT2PI;
    }
  }

  @Override
  public double pdf(VectorWritable vw) {
    return Math.exp(-(sumXminusCdivRsquared(vw.get()) / 2)) / zProd2piR;
  }
  
  /**
   * @param x
   *          a Vector
   * @return the zSum(((x[i]-c[i])/r[i])^2) over all i
   */
  private double sumXminusCdivRsquared(Vector x) {
    double result = 0;
    for (Iterator<Element> it = getRadius().iterateNonZero(); it.hasNext();) {
      Element radiusElem = it.next();
      int index = radiusElem.index();
      double quotient = (x.get(index) - getCenter().get(index))
          / radiusElem.get();
      result += quotient * quotient;
    }
    return result;
  }
  
}
