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

import java.util.Locale;

import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.math.Vector;

/**
 * 
 *@deprecated Use GaussianCluster instead
 */
public class SampledNormalModel extends NormalModel {
  
  public SampledNormalModel() {
  }
  
  public SampledNormalModel(int id, Vector mean, double sd) {
    super(id, mean, sd);
  }
  
  @Override
  public String toString() {
    return asFormatString(null);
  }
  
  /**
   * Return an instance with the same parameters
   * 
   * @return an SampledNormalModel
   */
  @Override
  public NormalModel sampleFromPosterior() {
    return new SampledNormalModel(getId(), getMean(), getStdDev());
  }
  
  @Override
  public String asFormatString(String[] bindings) {
    StringBuilder buf = new StringBuilder();
    buf.append("snm{n=").append(getS0()).append(" m=");
    if (getMean() != null) {
      buf.append(AbstractCluster.formatVector(getMean(), bindings));
    }
    buf.append(" sd=").append(String.format(Locale.ENGLISH, "%.2f", getStdDev())).append('}');
    return buf.toString();
  }
}
