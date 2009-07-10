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

import org.apache.mahout.matrix.Vector;

public class SampledNormalModel extends NormalModel {

  public SampledNormalModel() {
    super();
  }

  public SampledNormalModel(Vector mean, double sd) {
    super(mean, sd);
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append("snm{n=").append(s0).append(" m=[");
    if (mean != null) {
      for (int i = 0; i < mean.size(); i++) {
        buf.append(String.format("%.2f", mean.get(i))).append(", ");
      }
    }
    buf.append("] sd=").append(String.format("%.2f", sd)).append('}');
    return buf.toString();
  }

  /**
   * Return an instance with the same parameters
   *
   * @return an SampledNormalModel
   */
  @Override
  public NormalModel sample() {
    return new SampledNormalModel(mean, sd);
  }
}
