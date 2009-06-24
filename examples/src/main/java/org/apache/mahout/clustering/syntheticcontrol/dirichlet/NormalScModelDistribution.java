package org.apache.mahout.clustering.syntheticcontrol.dirichlet;

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
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.ModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.NormalModel;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;

/**
 * An implementation of the ModelDistribution interface suitable for testing the
 * DirichletCluster algorithm. Uses a Normal Distribution
 */
public class NormalScModelDistribution implements ModelDistribution<Vector> {

  @Override
  public Model<Vector>[] sampleFromPrior(int howMany) {
    Model<Vector>[] result = new NormalModel[howMany];
    for (int i = 0; i < howMany; i++) {
      DenseVector mean = new DenseVector(60);
      for (int j = 0; j < 60; j++)
        mean.set(j, UncommonDistributions.rNorm(30, 0.5));
      result[i] = new NormalModel(mean, 1);
    }
    return result;
  }

  @Override
  public Model<Vector>[] sampleFromPosterior(Model<Vector>[] posterior) {
    Model<Vector>[] result = new NormalModel[posterior.length];
    for (int i = 0; i < posterior.length; i++) {
      NormalModel m = (NormalModel) posterior[i];
      result[i] = m.sample();
    }
    return result;
  }
}
