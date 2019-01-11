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


import java.util.Arrays;
import java.util.List;

import org.apache.mahout.clustering.ClusteringUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.FastProjectionSearch;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.apache.mahout.math.neighborhood.Searcher;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.runners.Parameterized.Parameters;


@RunWith(Parameterized.class)
public class StreamingKMeansTest {
  private static final int NUM_DATA_POINTS = 1 << 16;
  private static final int NUM_DIMENSIONS = 6;
  private static final int NUM_PROJECTIONS = 2;
  private static final int SEARCH_SIZE = 10;

  private static Pair<List<Centroid>, List<Centroid>> syntheticData ;

  @Before
  public void setUp() {
    RandomUtils.useTestSeed();
    syntheticData =
      DataUtils.sampleMultiNormalHypercube(NUM_DIMENSIONS, NUM_DATA_POINTS);
  }

  private UpdatableSearcher searcher;
  private boolean allAtOnce;

  public StreamingKMeansTest(UpdatableSearcher searcher, boolean allAtOnce) {
    this.searcher = searcher;
    this.allAtOnce = allAtOnce;
  }

  @Parameters
  public static List<Object[]> generateData() {
    return Arrays.asList(new Object[][] {
        {new ProjectionSearch(new SquaredEuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE), true},
        {new FastProjectionSearch(new SquaredEuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE),
            true},
        {new ProjectionSearch(new SquaredEuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE), false},
        {new FastProjectionSearch(new SquaredEuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE),
            false},
    });
  }

  @Test
  public void testAverageDistanceCutoff() {
    double avgDistanceCutoff = 0;
    double avgNumClusters = 0;
    int numTests = 1;
    System.out.printf("Distance cutoff for %s\n", searcher.getClass().getName());
    for (int i = 0; i < numTests; ++i) {
      searcher.clear();
      int numStreamingClusters = (int)Math.log(syntheticData.getFirst().size()) * (1 <<
          NUM_DIMENSIONS);
      double distanceCutoff = 1.0e-6;
      double estimatedCutoff = ClusteringUtils.estimateDistanceCutoff(syntheticData.getFirst(),
          searcher.getDistanceMeasure(), 100);
      System.out.printf("[%d] Generated synthetic data [magic] %f [estimate] %f\n", i, distanceCutoff, estimatedCutoff);
      StreamingKMeans clusterer =
          new StreamingKMeans(searcher, numStreamingClusters, estimatedCutoff);
      clusterer.cluster(syntheticData.getFirst());
      avgDistanceCutoff += clusterer.getDistanceCutoff();
      avgNumClusters += clusterer.getNumClusters();
      System.out.printf("[%d] %f\n", i, clusterer.getDistanceCutoff());
    }
    avgDistanceCutoff /= numTests;
    avgNumClusters /= numTests;
    System.out.printf("Final: distanceCutoff: %f estNumClusters: %f\n", avgDistanceCutoff, avgNumClusters);
  }

  @Test
  public void testClustering() {
    searcher.clear();
    int numStreamingClusters = (int)Math.log(syntheticData.getFirst().size()) * (1 << NUM_DIMENSIONS);
    System.out.printf("k log n = %d\n", numStreamingClusters);
    double estimatedCutoff = ClusteringUtils.estimateDistanceCutoff(syntheticData.getFirst(),
        searcher.getDistanceMeasure(), 100);
    StreamingKMeans clusterer =
        new StreamingKMeans(searcher, numStreamingClusters, estimatedCutoff);

    long startTime = System.currentTimeMillis();
    if (allAtOnce) {
      clusterer.cluster(syntheticData.getFirst());
    } else {
      for (Centroid datapoint : syntheticData.getFirst()) {
        clusterer.cluster(datapoint);
      }
    }
    long endTime = System.currentTimeMillis();

    System.out.printf("%s %s\n", searcher.getClass().getName(), searcher.getDistanceMeasure()
        .getClass().getName());
    System.out.printf("Total number of clusters %d\n", clusterer.getNumClusters());

    System.out.printf("Weights: %f %f\n", ClusteringUtils.totalWeight(syntheticData.getFirst()),
        ClusteringUtils.totalWeight(clusterer));
    assertEquals("Total weight not preserved", ClusteringUtils.totalWeight(syntheticData.getFirst()),
        ClusteringUtils.totalWeight(clusterer), 1.0e-9);

    // and verify that each corner of the cube has a centroid very nearby
    double maxWeight = 0;
    for (Vector mean : syntheticData.getSecond()) {
      WeightedThing<Vector> v = searcher.search(mean, 1).get(0);
      maxWeight = Math.max(v.getWeight(), maxWeight);
    }
    assertTrue("Maximum weight too large " + maxWeight, maxWeight < 0.05);
    double clusterTime = (endTime - startTime) / 1000.0;
    System.out.printf("%s\n%.2f for clustering\n%.1f us per row\n\n",
        searcher.getClass().getName(), clusterTime,
        clusterTime / syntheticData.getFirst().size() * 1.0e6);

    // verify that the total weight of the centroids near each corner is correct
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
}
