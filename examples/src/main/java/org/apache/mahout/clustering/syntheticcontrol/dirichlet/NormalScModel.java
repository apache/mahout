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
package org.apache.mahout.clustering.syntheticcontrol.dirichlet;

import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.matrix.Vector;

public class NormalScModel implements Model<Vector> {

  private static final double sqrt2pi = Math.sqrt(2.0 * Math.PI);

  // the parameters
  public Vector mean;

  public double sd;

  // the observation statistics, initialized by the first observation
  int s0 = 0;

  Vector s1;

  Vector s2;

  public NormalScModel() {
  }

  public NormalScModel(Vector mean, double sd) {
    this.mean = mean;
    this.sd = sd;
    this.s0 = 0;
    this.s1 = mean.like();
    this.s2 = mean.like();
  }

  /**
   * Return an instance with the same parameters
   * @return an NormalModel
   */
  NormalScModel sample() {
    return new NormalScModel(mean, sd);
  }

  @Override
  public void observe(Vector x) {
    s0++;
    if (s1 == null)
      s1 = x;
    else
      s1 = s1.plus(x);
    if (s2 == null)
      s2 = x.times(x);
    else
      s2 = s2.plus(x.times(x));
  }

  @Override
  public void computeParameters() {
    if (s0 == 0)
      return;
    mean = s1.divide(s0);
    //TODO: is this the average of the 60 component stds??
    if (s0 > 1)
      sd = Math.sqrt(s2.times(s0).minus(s1.times(s1)).zSum() / (60 * 60)) / s0;
    else
      sd = Double.MIN_VALUE;
  }

  @Override
  // TODO: need to revisit this for reasonableness
  public double pdf(Vector x) {
    assert x.size() == 60;
    double sd2 = sd * sd;
    double exp = -(x.dot(x) - 2 * x.dot(mean) + mean.dot(mean)) / (2 * sd2);
    double ex = Math.exp(exp);
    return ex / (sd * sqrt2pi);
  }

  @Override
  public int count() {
    return s0;
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append("nm{n=").append(s0).append(" m=[");
    if (mean != null)
      for (int i = 0; i < mean.cardinality(); i++)
        buf.append(String.format("%.2f", mean.get(i))).append(", ");
    buf.append("] sd=").append(String.format("%.2f", sd)).append('}');
    return buf.toString();
  }
}
