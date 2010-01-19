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

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * An implementation of the ModelDistribution interface suitable for testing the DirichletCluster algorithm. Uses a
 * Normal Distribution
 */
public class SparseNormalModelDistribution extends VectorModelDistribution {

  public SparseNormalModelDistribution(VectorWritable modelPrototype) {
    super(modelPrototype);
  }

  public SparseNormalModelDistribution() {
    super();
  }

  @Override
  public Model<VectorWritable>[] sampleFromPrior(int howMany) {
    Model<VectorWritable>[] result = new NormalModel[howMany];
    for (int i = 0; i < howMany; i++) {
      Vector prototype = getModelPrototype().get();
      result[i] = new NormalModel(prototype.like(), 1);
    }
    return result;
  }

  @Override
  public Model<VectorWritable>[] sampleFromPosterior(Model<VectorWritable>[] posterior) {
    Model<VectorWritable>[] result = new NormalModel[posterior.length];
    for (int i = 0; i < posterior.length; i++) {
      NormalModel m = ((NormalModel) posterior[i]).sample();
      // trim insignificant mean elements from the posterior model to save sparse vector space
      for (int j = 0; j < m.getMean().size(); j++)
        if (Math.abs(m.getMean().get(j)) < 0.001)
          m.getMean().set(j, 0.0d);
      result[i] = m;
    }
    return result;
  }
}
