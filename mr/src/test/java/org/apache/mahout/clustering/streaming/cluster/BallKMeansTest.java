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
import org.apache.mahout.clustering.ClusteringUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.ConstantVector;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.Searcher;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.random.MultiNormal;
import org.apache.mahout.math.random.WeightedThing;
import org.apache.mahout.math.stats.OnlineSummarizer;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.apache.mahout.clustering.ClusteringUtils.totalWeight;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class BallKMeansTest {
  private static final int NUM_DATA_POINTS = 10000;
  private static final int NUM_DIMENSIONS = 4;
  private static final int NUM_ITERATIONS = 20;
  private static final double DISTRIBUTION_RADIUS = 0.01;

  @BeforeClass
  public static void setUp() {
    RandomUtils.useTestSeed();
    syntheticData = DataUtils.sampleMultiNormalHypercube(NUM_DIMENSIONS, NUM_DATA_POINTS, DISTRIBUTION_RADIUS);

  }

  private static Pair<List<Centroid>, List<Centroid>> syntheticData;
  private static final int K1 = 100;


  @Test
  public void testClusteringMultipleRuns() {
    for (int i = 1; i <= 10; ++i) {
      BallKMeans clusterer = new BallKMeans(new BruteSearch(new SquaredEuclideanDistanceMeasure()),
          1 << NUM_DIMENSIONS, NUM_ITERATIONS, true, i);
      clusterer.cluster(syntheticData.getFirst());
      double costKMeansPlusPlus = ClusteringUtils.totalClusterCost(syntheticData.getFirst(), clusterer);

      clusterer = new BallKMeans(new BruteSearch(new SquaredEuclideanDistanceMeasure()),
          1 << NUM_DIMENSIONS, NUM_ITERATIONS, false, i);
      clusterer.cluster(syntheticData.getFirst());
      double costKMeansRandom = ClusteringUtils.totalClusterCost(syntheticData.getFirst(), clusterer);

      System.out.printf("%d runs; kmeans++: %f; random: %f\n", i, costKMeansPlusPlus, costKMeansRandom);
      assertTrue("kmeans++ cost should be less than random cost", costKMeansPlusPlus < costKMeansRandom);
    }
  }

  @Test
  public void testClustering() {
    UpdatableSearcher searcher = new BruteSearch(new SquaredEuclideanDistanceMeasure());
    BallKMeans clusterer = new BallKMeans(searcher, 1 << NUM_DIMENSIONS, NUM_ITERATIONS);

    long startTime = System.currentTimeMillis();
    Pair<List<Centroid>, List<Centroid>> data = syntheticData;
    clusterer.cluster(data.getFirst());
    long endTime = System.currentTimeMillis();

    long hash = 0;
    for (Centroid centroid : data.getFirst()) {
      for (Vector.Element element : centroid.all()) {
        hash = 31 * hash + 17 * element.index() + Double.toHexString(element.get()).hashCode();
      }
    }
    System.out.printf("Hash = %08x\n", hash);

    assertEquals("Total weight not preserved", totalWeight(syntheticData.getFirst()), totalWeight(clusterer), 1.0e-9);

    // Verify that each corner of the cube has a centroid very nearby.
    // This is probably FALSE for large-dimensional spaces!
    OnlineSummarizer summarizer = new OnlineSummarizer();
    for (Vector mean : syntheticData.getSecond()) {
      WeightedThing<Vector> v = searcher.search(mean, 1).get(0);
      summarizer.add(v.getWeight());
    }
    assertTrue(String.format("Median weight [%f] too large [>%f]", summarizer.getMedian(),
        DISTRIBUTION_RADIUS), summarizer.getMedian() < DISTRIBUTION_RADIUS);

    double clusterTime = (endTime - startTime) / 1000.0;
    System.out.printf("%s\n%.2f for clustering\n%.1f us per row\n\n",
        searcher.getClass().getName(), clusterTime,
        clusterTime / syntheticData.getFirst().size() * 1.0e6);

    // Verify that the total weight of the centroids near each corner is correct.
    double[] cornerWeights = new double[1 << NUM_DIMENSIONS];
    Searcher trueFinder = new BruteSearch(new EuclideanDistanceMeasure());
    for (Vector trueCluster : syntheticData.getSecond()) {
      trueFinder.add(trueCluster);
    }
    for (Centroid centroid : clusterer) {
      WeightedThing<Vector> closest = trueFinder.search(centroid, 1).get(0);
      cornerWeights[((Centroid)closest.getValue()).getIndex()] += centroid.getWeight();
    }
    int expectedNumPoints = NUM_DATA_POINTS / (1 << NUM_DIMENSIONS);
    for (double v : cornerWeights) {
      System.out.printf("%f ", v);
    }
    System.out.println();
    for (double v : cornerWeights) {
      assertEquals(expectedNumPoints, v, 0);
    }
  }

  @Test
  public void testInitialization() {
    // Start with super clusterable data.
    List<? extends WeightedVector> data = cubishTestData(0.01);

    // Just do initialization of ball k-means. This should drop a point into each of the clusters.
    BallKMeans r = new BallKMeans(new BruteSearch(new SquaredEuclideanDistanceMeasure()), 6, 20);
    r.cluster(data);

    // Put the centroids into a matrix.
    Matrix x = new DenseMatrix(6, 5);
    int row = 0;
    for (Centroid c : r) {
      x.viewRow(row).assign(c.viewPart(0, 5));
      row++;
    }

    // Verify that each column looks right. Should contain zeros except for a single 6.
    final Vector columnNorms = x.aggregateColumns(new VectorFunction() {
      @Override
      public double apply(Vector f) {
        // Return the sum of three discrepancy measures.
        return Math.abs(f.minValue()) + Math.abs(f.maxValue() - 6) + Math.abs(f.norm(1) - 6);
      }
    });
    // Verify all errors are nearly zero.
    assertEquals(0, columnNorms.norm(1) / columnNorms.size(), 0.1);

    // Verify that the centroids are a permutation of the original ones.
    SingularValueDecomposition svd = new SingularValueDecomposition(x);
    Vector s = svd.getS().viewDiagonal().assign(Functions.div(6));
    assertEquals(5, s.getLengthSquared(), 0.05);
    assertEquals(5, s.norm(1), 0.05);
  }

  private static List<? extends WeightedVector> cubishTestData(double radius) {
    List<WeightedVector> data = Lists.newArrayListWithCapacity(K1 + 5000);
    int row = 0;

    MultiNormal g = new MultiNormal(radius, new ConstantVector(0, 10));
    for (int i = 0; i < K1; i++) {
      data.add(new WeightedVector(g.sample(), 1, row++));
    }

    for (int i = 0; i < 5; i++) {
      Vector m = new DenseVector(10);
      m.set(i, 6); // This was originally i == 0 ? 6 : 6 which can't be right
      MultiNormal gx = new MultiNormal(radius, m);
      for (int j = 0; j < 1000; j++) {
        data.add(new WeightedVector(gx.sample(), 1, row++));
      }
    }
    return data;
  }
}
