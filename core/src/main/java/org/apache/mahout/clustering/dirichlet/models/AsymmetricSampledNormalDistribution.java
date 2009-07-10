package org.apache.mahout.clustering.dirichlet.models;

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

import org.apache.mahout.clustering.dirichlet.UncommonDistributions;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;

/**
 * An implementation of the ModelDistribution interface suitable for testing the DirichletCluster algorithm. Uses a
 * Normal Distribution to sample the prior model values. Model values have a vector standard deviation, allowing
 * assymetrical regions to be covered by a model.
 */
public class AsymmetricSampledNormalDistribution implements
    ModelDistribution<Vector> {

  @Override
  public Model<Vector>[] sampleFromPrior(int howMany) {
    Model<Vector>[] result = new AsymmetricSampledNormalModel[howMany];
    for (int i = 0; i < howMany; i++) {
      double[] m = {UncommonDistributions.rNorm(0, 1),
          UncommonDistributions.rNorm(0, 1)};
      DenseVector mean = new DenseVector(m);
      double[] s = {UncommonDistributions.rNorm(1, 1),
          UncommonDistributions.rNorm(1, 1)};
      DenseVector sd = new DenseVector(s);
      result[i] = new AsymmetricSampledNormalModel(mean, sd);
    }
    return result;
  }

  @Override
  public Model<Vector>[] sampleFromPosterior(Model<Vector>[] posterior) {
    Model<Vector>[] result = new AsymmetricSampledNormalModel[posterior.length];
    for (int i = 0; i < posterior.length; i++) {
      AsymmetricSampledNormalModel m = (AsymmetricSampledNormalModel) posterior[i];
      result[i] = m.sample();
    }
    return result;
  }

}
