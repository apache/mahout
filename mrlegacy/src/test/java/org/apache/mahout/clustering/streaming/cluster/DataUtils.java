/*
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

package org.apache.mahout.clustering.streaming.cluster;

import java.util.List;

import com.google.common.collect.Lists;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.MultiNormal;

/**
 * A collection of miscellaneous utility functions for working with data to be clustered.
 * Includes methods for generating synthetic data and estimating distance cutoff.
 */
public final class DataUtils {
  private DataUtils() {
  }

  /**
   * Samples numDatapoints vectors of numDimensions cardinality centered around the vertices of a
   * numDimensions order hypercube. The distribution of points around these vertices is
   * multinormal with a radius of distributionRadius.
   * A hypercube of numDimensions has 2^numDimensions vertices. Keep this in mind when clustering
   * the data.
   *
   * Note that it is almost always the case that you want to call RandomUtils.useTestSeed() before
   * generating test data.  This means that you can't generate data in the declaration of a static
   * variable because such initializations happen before any @BeforeClass or @Before setup methods
   * are called.
   *
   *
   * @param numDimensions number of dimensions of the vectors to be generated.
   * @param numDatapoints number of data points to be generated.
   * @param distributionRadius radius of the distribution around the hypercube vertices.
   * @return a pair of lists, whose first element is the sampled points and whose second element
   * is the list of hypercube vertices that are the means of each distribution.
   */
  public static Pair<List<Centroid>, List<Centroid>> sampleMultiNormalHypercube(
      int numDimensions, int numDatapoints, double distributionRadius) {
    int pow2N = 1 << numDimensions;
    // Construct data samplers centered on the corners of a unit hypercube.
    // Additionally, keep the means of the distributions that will be generated so we can compare
    // these to the ideal cluster centers.
    List<Centroid> mean = Lists.newArrayListWithCapacity(pow2N);
    List<MultiNormal> rowSamplers = Lists.newArrayList();
    for (int i = 0; i < pow2N; i++) {
      Vector v = new DenseVector(numDimensions);
      // Select each of the num
      int pow2J = 1 << (numDimensions - 1);
      for (int j = 0; j < numDimensions; ++j) {
        v.set(j, 1.0 / pow2J * (i & pow2J));
        pow2J >>= 1;
      }
      mean.add(new Centroid(i, v, 1));
      rowSamplers.add(new MultiNormal(distributionRadius, v));
    }

    // Sample the requested number of data points.
    List<Centroid> data = Lists.newArrayListWithCapacity(numDatapoints);
    for (int i = 0; i < numDatapoints; ++i) {
      data.add(new Centroid(i, rowSamplers.get(i % pow2N).sample(), 1));
    }
    return new Pair<List<Centroid>, List<Centroid>>(data, mean);
  }

  /**
   * Calls sampleMultinormalHypercube(numDimension, numDataPoints, 0.01).
   * @see DataUtils#sampleMultiNormalHypercube(int, int, double)
   */
  public static Pair<List<Centroid>, List<Centroid>> sampleMultiNormalHypercube(int numDimensions,
                                                                                int numDatapoints) {
    return sampleMultiNormalHypercube(numDimensions, numDatapoints, 0.01);
  }
}
