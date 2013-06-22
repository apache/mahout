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

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import org.apache.mahout.clustering.ClusteringUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.random.Multinomial;
import org.apache.mahout.math.random.WeightedThing;

/**
 * Implements a ball k-means algorithm for weighted vectors with probabilistic seeding similar to k-means++.
 * The idea is that k-means++ gives good starting clusters and ball k-means can tune up the final result very nicely
 * in only a few passes (or even in a single iteration for well-clusterable data).
 *
 * A good reference for this class of algorithms is "The Effectiveness of Lloyd-Type Methods for the k-Means Problem"
 * by Rafail Ostrovsky, Yuval Rabani, Leonard J. Schulman and Chaitanya Swamy.  The code here uses the seeding strategy
 * as described in section 4.1.1 of that paper and the ball k-means step as described in section 4.2.  We support
 * multiple iterations in contrast to the algorithm described in the paper.
 */
public class BallKMeans implements Iterable<Centroid> {
  /**
   * The searcher containing the centroids.
   */
  private final UpdatableSearcher centroids;

  /**
   * The number of clusters to cluster the data into.
   */
  private final int numClusters;

  /**
   * The maximum number of iterations of the algorithm to run waiting for the cluster assignments
   * to stabilize. If there are no changes in cluster assignment earlier, we can finish early.
   */
  private final int maxNumIterations;

  /**
   * When deciding which points to include in the new centroid calculation,
   * it's preferable to exclude outliers since it increases the rate of convergence.
   * So, we calculate the distance from each cluster to its closest neighboring cluster. When
   * evaluating the points assigned to a cluster, we compare the distance between the centroid to
   * the point with the distance between the centroid and its closest centroid neighbor
   * multiplied by this trimFraction. If the distance between the centroid and the point is
   * greater, we consider it an outlier and we don't use it.
   */
  private final double trimFraction;

  /**
   * Selecting the initial centroids is the most important part of the ball k-means clustering. Poor choices, like two
   * centroids in the same actual cluster result in a low-quality final result.
   * k-means++ initialization yields good quality clusters, especially when using BallKMeans after StreamingKMeans as
   * the points have weights.
   * Simple, random selection of the points based on their weights is faster but sometimes fails to produce the
   * desired number of clusters.
   * This field is true if the initialization should be done with k-means++.
   */
  private final boolean kMeansPlusPlusInit;

  /**
   * When using trimFraction, the weight of each centroid will not be the sum of the weights of
   * the vectors assigned to that cluster because outliers are not used to compute the updated
   * centroid.
   * So, the total weight is probably wrong. This can be fixed by doing another pass over the
   * data points and adjusting the weights of each centroid. This doesn't update the coordinates
   * of the centroids, but is useful if the weights matter.
   */
  private final boolean correctWeights;

  /**
   * When running multiple ball k-means passes to get the one with the smallest total cost, can compute the
   * overall cost, using all the points for clustering, or reserve a fraction of them, testProbability in a test set.
   * The cost is the sum of the distances between each point and its corresponding centroid.
   * We then use this set of points to compute the total cost on. We're therefore trying to select the clustering
   * that best describes the underlying distribution of the clusters.
   * This field is the probability of assigning a given point to the test set. If this is 0, the cost will be computed
   * on the entire set of points.
   */
  private final double testProbability;

  /**
   * Whether or not testProbability > 0, i.e., there exists a non-empty 'test' set.
   */
  private final boolean splitTrainTest;

  /**
   * How many k-means runs to have. If there's more than one run, we compute the cost of each clustering as described
   * above and select the clustering that minimizes the cost.
   * Multiple runs are a lot more useful when using the random initialization. With kmeans++, 1-2 runs are enough and
   * more runs are not likely to help quality much.
   */
  private final int numRuns;

  /**
   * Random object to sample values from.
   */
  private final Random random;

  public BallKMeans(UpdatableSearcher searcher, int numClusters, int maxNumIterations) {
    // By default, the trimFraction is 0.9, k-means++ is used, the weights will be corrected at the end,
    // there will be 0 points in the test set and 1 run.
    this(searcher, numClusters, maxNumIterations, 0.9, true, true, 0.0, 1);
  }

  public BallKMeans(UpdatableSearcher searcher, int numClusters, int maxNumIterations,
                    boolean kMeansPlusPlusInit, int numRuns) {
    // By default, the trimFraction is 0.9, k-means++ is used, the weights will be corrected at the end,
    // there will be 10% points of in the test set.
    this(searcher, numClusters, maxNumIterations, 0.9, kMeansPlusPlusInit, true, 0.1, numRuns);
  }

  public BallKMeans(UpdatableSearcher searcher, int numClusters, int maxNumIterations,
                    double trimFraction, boolean kMeansPlusPlusInit, boolean correctWeights,
                    double testProbability, int numRuns) {
    Preconditions.checkArgument(searcher.size() == 0, "Searcher must be empty initially to populate with centroids");
    Preconditions.checkArgument(numClusters > 0, "The requested number of clusters must be positive");
    Preconditions.checkArgument(maxNumIterations > 0, "The maximum number of iterations must be positive");
    Preconditions.checkArgument(trimFraction > 0, "The trim fraction must be positive");
    Preconditions.checkArgument(testProbability >= 0 && testProbability < 1, "The testProbability must be in [0, 1)");
    Preconditions.checkArgument(numRuns > 0, "There has to be at least one run");

    this.centroids = searcher;
    this.numClusters = numClusters;
    this.maxNumIterations = maxNumIterations;

    this.trimFraction = trimFraction;
    this.kMeansPlusPlusInit = kMeansPlusPlusInit;
    this.correctWeights = correctWeights;

    this.testProbability = testProbability;
    this.splitTrainTest = testProbability > 0;
    this.numRuns = numRuns;

    this.random = RandomUtils.getRandom();
  }

  public Pair<List<? extends WeightedVector>, List<? extends WeightedVector>> splitTrainTest(
      List<? extends WeightedVector> datapoints) {
    // If there will be no points assigned to the test set, return now.
    if (testProbability == 0) {
      return new Pair<List<? extends WeightedVector>, List<? extends WeightedVector>>(datapoints,
          Lists.<WeightedVector>newArrayList());
    }

    int numTest = (int) (testProbability * datapoints.size());
    Preconditions.checkArgument(numTest > 0 && numTest < datapoints.size(),
        "Must have nonzero number of training and test vectors. Asked for %.1f %% of %d vectors for test",
        testProbability * 100, datapoints.size());

    Collections.shuffle(datapoints);
    return new Pair<List<? extends WeightedVector>, List<? extends WeightedVector>>(
        datapoints.subList(numTest, datapoints.size()), datapoints.subList(0, numTest));
  }

  /**
   * Clusters the datapoints in the list doing either random seeding of the centroids or k-means++.
   *
   * @param datapoints the points to be clustered.
   * @return an UpdatableSearcher with the resulting clusters.
   */
  public UpdatableSearcher cluster(List<? extends WeightedVector> datapoints) {
    Pair<List<? extends WeightedVector>, List<? extends WeightedVector>> trainTestSplit = splitTrainTest(datapoints);
    List<Vector> bestCentroids = Lists.newArrayList();
    double cost = Double.POSITIVE_INFINITY;
    double bestCost = Double.POSITIVE_INFINITY;
    for (int i = 0; i < numRuns; ++i) {
      centroids.clear();
      if (kMeansPlusPlusInit) {
        // Use k-means++ to set initial centroids.
        initializeSeedsKMeansPlusPlus(trainTestSplit.getFirst());
      } else {
        // Randomly select the initial centroids.
        initializeSeedsRandomly(trainTestSplit.getFirst());
      }
      // Do k-means iterations with trimmed mean computation (aka ball k-means).
      if (numRuns > 1) {
        // If the clustering is successful (there are no zero-weight centroids).
        iterativeAssignment(trainTestSplit.getFirst());
        // Compute the cost of the clustering and possibly save the centroids.
        cost = ClusteringUtils.totalClusterCost(
            splitTrainTest ? datapoints : trainTestSplit.getSecond(), centroids);
        if (cost < bestCost) {
          bestCost = cost;
          bestCentroids.clear();
          Iterables.addAll(bestCentroids, centroids);
        }
      } else {
        // If there is only going to be one run, the cost doesn't need to be computed, so we just return the clustering.
        iterativeAssignment(datapoints);
        return centroids;
      }
    }
    if (bestCost == Double.POSITIVE_INFINITY) {
      throw new RuntimeException("No valid clustering was found");
    }
    if (cost != bestCost) {
      centroids.clear();
      centroids.addAll(bestCentroids);
    }
    if (correctWeights) {
      for (WeightedVector testDatapoint : trainTestSplit.getSecond()) {
        WeightedVector closest = (WeightedVector) centroids.searchFirst(testDatapoint, false).getValue();
        closest.setWeight(closest.getWeight() + testDatapoint.getWeight());
      }
    }
    return centroids;
  }

  /**
   * Selects some of the original points randomly with probability proportional to their weights. This is much
   * less sophisticated than the kmeans++ approach, however it is faster and coupled with
   *
   * The side effect of this method is to fill the centroids structure itself.
   *
   * @param datapoints The datapoints to select from.  These datapoints should be WeightedVectors of some kind.
   */
  private void initializeSeedsRandomly(List<? extends WeightedVector> datapoints) {
    int numDatapoints = datapoints.size();
    double totalWeight = 0;
    for (WeightedVector datapoint : datapoints) {
      totalWeight += datapoint.getWeight();
    }
    Multinomial<Integer> seedSelector = new Multinomial<Integer>();
    for (int i = 0; i < numDatapoints; ++i) {
      seedSelector.add(i, datapoints.get(i).getWeight() / totalWeight);
    }
    for (int i = 0; i < numClusters; ++i) {
      int sample = seedSelector.sample();
      seedSelector.delete(sample);
      Centroid centroid = new Centroid(datapoints.get(sample));
      centroid.setIndex(i);
      centroids.add(centroid);
    }
  }

  /**
   * Selects some of the original points according to the k-means++ algorithm.  The basic idea is that
   * points are selected with probability proportional to their distance from any selected point.  In
   * this version, points have weights which multiply their likelihood of being selected.  This is the
   * same as if there were as many copies of the same point as indicated by the weight.
   *
   * This is pretty expensive, but it vastly improves the quality and convergences of the k-means algorithm.
   * The basic idea can be made much faster by only processing a random subset of the original points.
   * In the context of streaming k-means, the total number of possible seeds will be about k log n so this
   * selection will cost O(k^2 (log n)^2) which isn't much worse than the random sampling idea.  At
   * n = 10^9, the cost of this initialization will be about 10x worse than a reasonable random sampling
   * implementation.
   *
   * The side effect of this method is to fill the centroids structure itself.
   *
   * @param datapoints The datapoints to select from.  These datapoints should be WeightedVectors of some kind.
   */
  private void initializeSeedsKMeansPlusPlus(List<? extends WeightedVector> datapoints) {
    Preconditions.checkArgument(datapoints.size() > 1, "Must have at least two datapoints points to cluster " +
        "sensibly");
    Preconditions.checkArgument(datapoints.size() >= numClusters,
        String.format("Must have more datapoints [%d] than clusters [%d]", datapoints.size(), numClusters));
    // Compute the centroid of all of the datapoints.  This is then used to compute the squared radius of the datapoints.
    Centroid center = new Centroid(datapoints.iterator().next());
    for (WeightedVector row : Iterables.skip(datapoints, 1)) {
      center.update(row);
    }

    // Given the centroid, we can compute \Delta_1^2(X), the total squared distance for the datapoints
    // this accelerates seed selection.
    double deltaX = 0;
    DistanceMeasure distanceMeasure = centroids.getDistanceMeasure();
    for (WeightedVector row : datapoints) {
      deltaX += distanceMeasure.distance(row, center);
    }

    // Find the first seed c_1 (and conceptually the second, c_2) as might be done in the 2-means clustering so that
    // the probability of selecting c_1 and c_2 is proportional to || c_1 - c_2 ||^2.  This is done
    // by first selecting c_1 with probability:
    //
    // p(c_1) = sum_{c_1} || c_1 - c_2 ||^2 \over sum_{c_1, c_2} || c_1 - c_2 ||^2
    //
    // This can be simplified to:
    //
    // p(c_1) = \Delta_1^2(X) + n || c_1 - c ||^2 / (2 n \Delta_1^2(X))
    //
    // where c = \sum x / n and \Delta_1^2(X) = sum || x - c ||^2
    //
    // All subsequent seeds c_i (including c_2) can then be selected from the remaining points with probability
    // proportional to Pr(c_i == x_j) = min_{m < i} || c_m - x_j ||^2.

    // Multinomial distribution of vector indices for the selection seeds. These correspond to
    // the indices of the vectors in the original datapoints list.
    Multinomial<Integer> seedSelector = new Multinomial<Integer>();
    for (int i = 0; i < datapoints.size(); ++i) {
      double selectionProbability =
          deltaX + datapoints.size() * distanceMeasure.distance(datapoints.get(i), center);
      seedSelector.add(i, selectionProbability);
    }

    int selected = random.nextInt(datapoints.size());
    Centroid c_1 = new Centroid(datapoints.get(selected).clone());
    c_1.setIndex(0);
    // Construct a set of weighted things which can be used for random selection.  Initial weights are
    // set to the squared distance from c_1
    for (int i = 0; i < datapoints.size(); ++i) {
      WeightedVector row = datapoints.get(i);
      double w = distanceMeasure.distance(c_1, row) * 2 * Math.log(1 + row.getWeight());
      seedSelector.set(i, w);
    }

    // From here, seeds are selected with probability proportional to:
    //
    // r_i = min_{c_j} || x_i - c_j ||^2
    //
    // when we only have c_1, we have already set these distances and as we select each new
    // seed, we update the minimum distances.
    centroids.add(c_1);
    int clusterIndex = 1;
    while (centroids.size() < numClusters) {
      // Select according to weights.
      int seedIndex = seedSelector.sample();
      Centroid nextSeed = new Centroid(datapoints.get(seedIndex));
      nextSeed.setIndex(clusterIndex++);
      centroids.add(nextSeed);
      // Don't select this one again.
      seedSelector.delete(seedIndex);
      // Re-weight everything according to the minimum distance to a seed.
      for (int currSeedIndex : seedSelector) {
        WeightedVector curr = datapoints.get(currSeedIndex);
        double newWeight = nextSeed.getWeight() * distanceMeasure.distance(nextSeed, curr);
        if (newWeight < seedSelector.getWeight(currSeedIndex)) {
          seedSelector.set(currSeedIndex, newWeight);
        }
      }
    }
  }

  /**
   * Examines the datapoints and updates cluster centers to be the centroid of the nearest datapoints points.  To
   * compute a new center for cluster c_i, we average all points that are closer than d_i * trimFraction
   * where d_i is
   *
   * d_i = min_j \sqrt ||c_j - c_i||^2
   *
   * By ignoring distant points, the centroids converge more quickly to a good approximation of the
   * optimal k-means solution (given good starting points).
   *
   * @param datapoints the points to cluster.
   */
  private void iterativeAssignment(List<? extends WeightedVector> datapoints) {
    DistanceMeasure distanceMeasure = centroids.getDistanceMeasure();
    // closestClusterDistances.get(i) is the distance from the i'th cluster to its closest
    // neighboring cluster.
    List<Double> closestClusterDistances = Lists.newArrayListWithExpectedSize(numClusters);
    // clusterAssignments[i] == j means that the i'th point is assigned to the j'th cluster. When
    // these don't change, we are done.
    // Each point is assigned to the invalid "-1" cluster initially.
    List<Integer> clusterAssignments = Lists.newArrayList(Collections.nCopies(datapoints.size(), -1));

    boolean changed = true;
    for (int i = 0; changed && i < maxNumIterations; i++) {
      changed = false;
      // We compute what the distance between each cluster and its closest neighbor is to set a
      // proportional distance threshold for points that should be involved in calculating the
      // centroid.
      closestClusterDistances.clear();
      for (Vector center : centroids) {
        // If a centroid has no points assigned to it, the clustering failed.
        Vector closestOtherCluster = centroids.searchFirst(center, true).getValue();
        closestClusterDistances.add(distanceMeasure.distance(center, closestOtherCluster));
      }

      // Copies the current cluster centroids to newClusters and sets their weights to 0. This is
      // so we calculate the new centroids as we go through the datapoints.
      List<Centroid> newCentroids = Lists.newArrayList();
      for (Vector centroid : centroids) {
        // need a deep copy because we will mutate these values
        Centroid newCentroid = (Centroid)centroid.clone();
        newCentroid.setWeight(0);
        newCentroids.add(newCentroid);
      }

      // Pass over the datapoints computing new centroids.
      for (int j = 0; j < datapoints.size(); ++j) {
        WeightedVector datapoint = datapoints.get(j);
        // Get the closest cluster this point belongs to.
        WeightedThing<Vector> closestPair = centroids.searchFirst(datapoint, false);
        int closestIndex = ((WeightedVector) closestPair.getValue()).getIndex();
        double closestDistance = closestPair.getWeight();
        // Update its cluster assignment if necessary.
        if (closestIndex != clusterAssignments.get(j)) {
          changed = true;
          clusterAssignments.set(j, closestIndex);
        }
        // Only update if the datapoints point is near enough. What this means is that the weight
        // of outliers is NOT taken into account and the final weights of the centroids will
        // reflect this (it will be less or equal to the initial sum of the weights).
        if (closestDistance < trimFraction * closestClusterDistances.get(closestIndex)) {
          newCentroids.get(closestIndex).update(datapoint);
        }
      }
      // Add the new centers back into searcher.
      centroids.clear();
      centroids.addAll(newCentroids);
    }

    if (correctWeights) {
      for (Vector v : centroids) {
        ((Centroid)v).setWeight(0);
      }
      for (WeightedVector datapoint : datapoints) {
        Centroid closestCentroid = (Centroid) centroids.searchFirst(datapoint, false).getValue();
        closestCentroid.setWeight(closestCentroid.getWeight() + datapoint.getWeight());
      }
    }
  }

  @Override
  public Iterator<Centroid> iterator() {
    return Iterators.transform(centroids.iterator(), new Function<Vector, Centroid>() {
      @Override
      public Centroid apply(Vector input) {
        Preconditions.checkArgument(input instanceof Centroid, "Non-centroid in centroids " +
            "searcher");
        //noinspection ConstantConditions
        return (Centroid)input;
      }
    });
  }
}
