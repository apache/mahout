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

import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.clustering.dirichlet.UncommonDistributions;
import org.apache.mahout.math.Vector;
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
    // return the product of the component pdfs
    // TODO: is this reasonable? correct? It seems to work in some cases.
    double pdf = 1;
    for (int i = 0; i < x.size(); i++) {
      // small prior on stdDev to avoid numeric instability when stdDev==0
      pdf *= UncommonDistributions.dNorm(x.getQuick(i),
          getCenter().getQuick(i), getRadius().getQuick(i) + 0.000001);
    }
    return pdf;
  }
  
}
