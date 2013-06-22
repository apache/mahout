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
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.jet.math.Constants;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.random.WeightedThing;

/**
 * Implements a streaming k-means algorithm for weighted vectors.
 * The goal clustering points one at a time, especially useful for MapReduce mappers that get inputs one at a time.
 *
 * A rough description of the algorithm:
 * Suppose there are l clusters at one point and a new point p is added.
 * The new point can either be added to one of the existing l clusters or become a new cluster. To decide:
 * - let c be the closest cluster to point p;
 * - let d be the distance between c and p;
 * - if d > distanceCutoff, create a new cluster from p (p is too far away from the clusters to be part of them;
 * distanceCutoff represents the largest distance from a point its assigned cluster's centroid);
 * - else (d <= distanceCutoff), create a new cluster with probability d / distanceCutoff (the probability of creating
 * a new cluster increases as d increases).
 * There will be either l points or l + 1 points after processing a new point.
 *
 * As the number of clusters increases, it will go over the numClusters limit (numClusters represents a recommendation
 * for the number of clusters that there should be at the end). To decrease the number of clusters the existing clusters
 * are treated as data points and are re-clustered (collapsed). This tends to make the number of clusters go down.
 * If the number of clusters is still too high, distanceCutoff is increased.
 *
 * For more details, see:
 * - "Streaming  k-means approximation" by N. Ailon, R. Jaiswal, C. Monteleoni
 * http://books.nips.cc/papers/files/nips22/NIPS2009_1085.pdf
 * - "Fast and Accurate k-means for Large Datasets" by M. Shindler, A. Wong, A. Meyerson,
 * http://books.nips.cc/papers/files/nips24/NIPS2011_1271.pdf
 */
public class StreamingKMeans implements Iterable<Centroid> {
  /**
   * The searcher containing the centroids that resulted from the clustering of points until now. When adding a new
   * point we either assign it to one of the existing clusters in this searcher or create a new centroid for it.
   */
  private final UpdatableSearcher centroids;

  /**
   * The estimated number of clusters to cluster the data in. If the actual number of clusters increases beyond this
   * limit, the clusters will be "collapsed" (re-clustered, by treating them as data points). This doesn't happen
   * recursively and a collapse might not necessarily make the number of actual clusters drop to less than this limit.
   *
   * If the goal is clustering a large data set into k clusters, numClusters SHOULD NOT BE SET to k. StreamingKMeans is
   * useful to reduce the size of the data set by the mappers so that it can fit into memory in one reducer that runs
   * BallKMeans.
   *
   * It is NOT MEANT to cluster the data into k clusters in one pass because it can't guarantee that there will in fact
   * be k clusters in total. This is because of the dynamic nature of numClusters over the course of the runtime.
   * To get an exact number of clusters, another clustering algorithm needs to be applied to the results.
   */
  private int numClusters;

  /**
   * The number of data points seen so far. This is important for re-estimating numClusters when deciding to collapse
   * the existing clusters.
   */
  private int numProcessedDatapoints = 0;

  /**
   * This is the current value of the distance cutoff.  Points which are much closer than this to a centroid will stick
   * to it almost certainly. Points further than this to any centroid will form a new cluster.
   *
   * This increases (is multiplied by beta) when a cluster collapse did not make the number of clusters drop to below
   * numClusters (it effectively increases the tolerance for cluster compactness discouraging the creation of new
   * clusters). Since a collapse only happens when centroids.size() > clusterOvershoot * numClusters, the cutoff
   * increases when the collapse didn't at least remove the slack in the number of clusters.
   */
  private double distanceCutoff;

  /**
   * Parameter that controls the growth of the distanceCutoff. After n increases of the
   * distanceCutoff starting at d_0, the final value is d_0 * beta^n (distance cutoffs increase following a geometric
   * progression with ratio beta).
   */
  private final double beta;

  /**
   * Multiplying clusterLogFactor with numProcessedDatapoints gets an estimate of the suggested
   * number of clusters. This mirrors the recommended number of clusters for n points where there should be k actual
   * clusters, k * log n. In the case of our estimate we use clusterLogFactor * log(numProcessedDataPoints).
   *
   * It is important to note that numClusters is NOT k. It is an estimate of k * log n.
   */
  private final double clusterLogFactor;

  /**
   * Centroids are collapsed when the number of clusters becomes greater than clusterOvershoot * numClusters. This
   * effectively means having a slack in numClusters so that the actual number of centroids, centroids.size() tracks
   * numClusters approximately. The idea is that the actual number of clusters should be at least numClusters but not
   * much more (so that we don't end up having 1 cluster / point).
   */
  private final double clusterOvershoot;

  /**
   * Random object to sample values from.
   */
  private final Random random = RandomUtils.getRandom();

  /**
   * Calls StreamingKMeans(searcher, numClusters, 1.3, 10, 2).
   * @see StreamingKMeans#StreamingKMeans(org.apache.mahout.math.neighborhood.UpdatableSearcher, int,
   * double, double, double, double)
   */
  public StreamingKMeans(UpdatableSearcher searcher, int numClusters) {
    this(searcher, numClusters, 1.0 / numClusters, 1.3, 20, 2);
  }

  /**
   * Calls StreamingKMeans(searcher, numClusters, distanceCutoff, 1.3, 10, 2).
   * @see StreamingKMeans#StreamingKMeans(org.apache.mahout.math.neighborhood.UpdatableSearcher, int,
   * double, double, double, double)
   */
  public StreamingKMeans(UpdatableSearcher searcher, int numClusters, double distanceCutoff) {
    this(searcher, numClusters, distanceCutoff, 1.3, 20, 2);
  }

  /**
   * Creates a new StreamingKMeans class given a searcher and the number of clusters to generate.
   *
   * @param searcher A Searcher that is used for performing nearest neighbor search. It MUST BE
   *                 EMPTY initially because it will be used to keep track of the cluster
   *                 centroids.
   * @param numClusters An estimated number of clusters to generate for the data points.
   *                    This can adjusted, but the actual number will depend on the data. The
   * @param distanceCutoff  The initial distance cutoff representing the value of the
   *                      distance between a point and its closest centroid after which
   *                      the new point will definitely be assigned to a new cluster.
   * @param beta Ratio of geometric progression to use when increasing distanceCutoff. After n increases, distanceCutoff
   *             becomes distanceCutoff * beta^n. A smaller value increases the distanceCutoff less aggressively.
   * @param clusterLogFactor Value multiplied with the number of points counted so far estimating the number of clusters
   *                         to aim for. If the final number of clusters is known and this clustering is only for a
   *                         sketch of the data, this can be the final number of clusters, k.
   * @param clusterOvershoot Multiplicative slack factor for slowing down the collapse of the clusters.
   */
  public StreamingKMeans(UpdatableSearcher searcher, int numClusters,
                         double distanceCutoff, double beta, double clusterLogFactor, double clusterOvershoot) {
    this.centroids = searcher;
    this.numClusters = numClusters;
    this.distanceCutoff = distanceCutoff;
    this.beta = beta;
    this.clusterLogFactor = clusterLogFactor;
    this.clusterOvershoot = clusterOvershoot;
  }

  /**
   * @return an Iterator to the Centroids contained in this clusterer.
   */
  @Override
  public Iterator<Centroid> iterator() {
    return Iterators.transform(centroids.iterator(), new Function<Vector, Centroid>() {
      @Override
      public Centroid apply(Vector input) {
        return (Centroid)input;
      }
    });
  }

  /**
   * Cluster the rows of a matrix, treating them as Centroids with weight 1.
   * @param data matrix whose rows are to be clustered.
   * @return the UpdatableSearcher containing the resulting centroids.
   */
  public UpdatableSearcher cluster(Matrix data) {
    return cluster(Iterables.transform(data, new Function<MatrixSlice, Centroid>() {
      @Override
      public Centroid apply(MatrixSlice input) {
        // The key in a Centroid is actually the MatrixSlice's index.
        return Centroid.create(input.index(), input.vector());
      }
    }));
  }

  /**
   * Cluster the data points in an Iterable<Centroid>.
   * @param datapoints Iterable whose elements are to be clustered.
   * @return the UpdatableSearcher containing the resulting centroids.
   */
  public UpdatableSearcher cluster(Iterable<Centroid> datapoints) {
    return clusterInternal(datapoints, false);
  }

  /**
   * Cluster one data point.
   * @param datapoint to be clustered.
   * @return the UpdatableSearcher containing the resulting centroids.
   */
  public UpdatableSearcher cluster(final Centroid datapoint) {
    return cluster(new Iterable<Centroid>() {
      @Override
      public Iterator<Centroid> iterator() {
        return new Iterator<Centroid>() {
          private boolean accessed = false;

          @Override
          public boolean hasNext() {
            return !accessed;
          }

          @Override
          public Centroid next() {
            accessed = true;
            return datapoint;
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException();
          }
        };
      }
    });
  }

  /**
   * @return the number of clusters computed from the points until now.
   */
  public int getNumClusters() {
    return centroids.size();
  }

  /**
   * Internal clustering method that gets called from the other wrappers.
   * @param datapoints Iterable of data points to be clustered.
   * @param collapseClusters whether this is an "inner" clustering and the datapoints are the previously computed
   *                         centroids. Some logic is different to ensure counters are consistent but it behaves
   *                         nearly the same.
   * @return the UpdatableSearcher containing the resulting centroids.
   */
  private UpdatableSearcher clusterInternal(Iterable<Centroid> datapoints, boolean collapseClusters) {
    Iterator<Centroid> datapointsIterator = datapoints.iterator();
    if (!datapointsIterator.hasNext()) {
      return centroids;
    }

    int oldNumProcessedDataPoints = numProcessedDatapoints;
    // We clear the centroids we have in case of cluster collapse, the old clusters are the
    // datapoints but we need to re-cluster them.
    if (collapseClusters) {
      centroids.clear();
      numProcessedDatapoints = 0;
    }

    if (centroids.size() == 0) {
      // Assign the first datapoint to the first cluster.
      // Adding a vector to a searcher would normally just reference the copy,
      // but we could potentially mutate it and so we need to make a clone.
      centroids.add(datapointsIterator.next().clone());
      ++numProcessedDatapoints;
    }

    // To cluster, we scan the data and either add each point to the nearest group or create a new group.
    // when we get too many groups, we need to increase the threshold and rescan our current groups
    while (datapointsIterator.hasNext()) {
      Centroid row = datapointsIterator.next();
      // Get the closest vector and its weight as a WeightedThing<Vector>.
      // The weight of the WeightedThing is the distance to the query and the value is a
      // reference to one of the vectors we added to the searcher previously.
      WeightedThing<Vector> closestPair = centroids.searchFirst(row, false);

      // We get a uniformly distributed random number between 0 and 1 and compare it with the
      // distance to the closest cluster divided by the distanceCutoff.
      // This is so that if the closest cluster is further than distanceCutoff,
      // closestPair.getWeight() / distanceCutoff > 1 which will trigger the creation of a new
      // cluster anyway.
      // However, if the ratio is less than 1, we want to create a new cluster with probability
      // proportional to the distance to the closest cluster.
      double sample = random.nextDouble();
      if (sample < row.getWeight() * closestPair.getWeight() / distanceCutoff) {
        // Add new centroid, note that the vector is copied because we may mutate it later.
        centroids.add(row.clone());
      } else {
        // Merge the new point with the existing centroid. This will update the centroid's actual
        // position.
        // We know that all the points we inserted in the centroids searcher are (or extend)
        // WeightedVector, so the cast will always succeed.
        Centroid centroid = (Centroid) closestPair.getValue();

        // We will update the centroid by removing it from the searcher and reinserting it to
        // ensure consistency.
        if (!centroids.remove(centroid, Constants.EPSILON)) {
          throw new RuntimeException("Unable to remove centroid");
        }
        centroid.update(row);
        centroids.add(centroid);

      }
      ++numProcessedDatapoints;

      if (!collapseClusters && centroids.size() > clusterOvershoot * numClusters) {
        numClusters = (int) Math.max(numClusters, clusterLogFactor * Math.log(numProcessedDatapoints));

        List<Centroid> shuffled = Lists.newArrayList();
        for (Vector vector : centroids) {
          shuffled.add((Centroid) vector);
        }
        Collections.shuffle(shuffled);
        // Re-cluster using the shuffled centroids as data points. The centroids member variable
        // is modified directly.
        clusterInternal(shuffled, true);

        if (centroids.size() > numClusters) {
          distanceCutoff *= beta;
        }
      }
    }

    if (collapseClusters) {
      numProcessedDatapoints = oldNumProcessedDataPoints;
    }
    return centroids;
  }

  public void reindexCentroids() {
    int numCentroids = 0;
    for (Centroid centroid : this) {
      centroid.setIndex(numCentroids++);
    }
  }

  /**
   * @return the distanceCutoff (an upper bound for the maximum distance within a cluster).
   */
  public double getDistanceCutoff() {
    return distanceCutoff;
  }

  public void setDistanceCutoff(double distanceCutoff) {
    this.distanceCutoff = distanceCutoff;
  }

  public DistanceMeasure getDistanceMeasure() {
    return centroids.getDistanceMeasure();
  }
}
