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

package org.apache.mahout.math.neighborhood;

import java.util.List;

import com.google.common.collect.Lists;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.ChineseRestaurant;
import org.apache.mahout.math.random.MultiNormal;
import org.apache.mahout.math.random.Sampler;

/**
 * Samples from clusters that have varying frequencies but constant radius.
 */
public class LumpyData implements Sampler<Vector> {
  // size of the clusters
  private final double radius;

  // figures out which cluster to look at
  private final Sampler<Integer> cluster;

  // remembers centroids of clusters
  private final List<Sampler<Vector>> centroids = Lists.newArrayList();

  // how the centroids are generated
  private final MultiNormal centers;

  /**
   * Samples from a lumpy distribution that acts a bit more like real data than just sampling from a normal distribution.
   * @param dimension   The dimension of the vectors to return.
   * @param radius      The size of the clusters we sample from.
   * @param alpha       Controls the growth of the number of clusters.  The number of clusters will be about alpha * log(samples)
   */
  public LumpyData(int dimension, double radius, double alpha) {
    this.centers = new MultiNormal(dimension);
    this.radius = radius;
    cluster = new ChineseRestaurant(alpha);
  }

  public static Matrix lumpyRandomData(int numDataPoints, int numDimensions) {
    final Matrix data = new DenseMatrix(numDataPoints, numDimensions);
    final LumpyData clusters = new LumpyData(numDimensions, 0.05, 10);
    for (MatrixSlice row : data) {
      row.vector().assign(clusters.sample());
    }
    return data;
  }

  @Override
  public Vector sample() {
    int id = cluster.sample();
    if (id >= centroids.size()) {
      // need to invent a new cluster
      centroids.add(new MultiNormal(radius, centers.sample()));
    }
    return centroids.get(id).sample();
  }
}
