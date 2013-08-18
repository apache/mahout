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

package org.apache.mahout.clustering;

import java.util.List;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.apache.mahout.math.neighborhood.Searcher;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.random.WeightedThing;
import org.apache.mahout.math.stats.OnlineSummarizer;

public final class ClusteringUtils {
  private ClusteringUtils() {
  }

  /**
   * Computes the summaries for the distances in each cluster.
   * @param datapoints iterable of datapoints.
   * @param centroids iterable of Centroids.
   * @return a list of OnlineSummarizers where the i-th element is the summarizer corresponding to the cluster whose
   * index is i.
   */
  public static List<OnlineSummarizer> summarizeClusterDistances(Iterable<? extends Vector> datapoints,
                                                                 Iterable<? extends Vector> centroids,
                                                                 DistanceMeasure distanceMeasure) {
    UpdatableSearcher searcher = new ProjectionSearch(distanceMeasure, 3, 1);
    searcher.addAll(centroids);
    List<OnlineSummarizer> summarizers = Lists.newArrayList();
    if (searcher.size() == 0) {
      return summarizers;
    }
    for (int i = 0; i < searcher.size(); ++i) {
      summarizers.add(new OnlineSummarizer());
    }
    for (Vector v : datapoints) {
      Centroid closest = (Centroid)searcher.search(v,  1).get(0).getValue();
      OnlineSummarizer summarizer = summarizers.get(closest.getIndex());
      summarizer.add(distanceMeasure.distance(v, closest));
    }
    return summarizers;
  }

  /**
   * Adds up the distances from each point to its closest cluster and returns the sum.
   * @param datapoints iterable of datapoints.
   * @param centroids iterable of Centroids.
   * @return the total cost described above.
   */
  public static double totalClusterCost(Iterable<? extends Vector> datapoints, Iterable<? extends Vector> centroids) {
    DistanceMeasure distanceMeasure = new EuclideanDistanceMeasure();
    UpdatableSearcher searcher = new ProjectionSearch(distanceMeasure, 3, 1);
    searcher.addAll(centroids);
    return totalClusterCost(datapoints, searcher);
  }

  /**
   * Adds up the distances from each point to its closest cluster and returns the sum.
   * @param datapoints iterable of datapoints.
   * @param centroids searcher of Centroids.
   * @return the total cost described above.
   */
  public static double totalClusterCost(Iterable<? extends Vector> datapoints, Searcher centroids) {
    double totalCost = 0;
    for (Vector vector : datapoints) {
      totalCost += centroids.searchFirst(vector, false).getWeight();
    }
    return totalCost;
  }

  /**
   * Estimates the distance cutoff. In StreamingKMeans, the distance between two vectors divided
   * by this value is used as a probability threshold when deciding whether to form a new cluster
   * or not.
   * Small values (comparable to the minimum distance between two points) are preferred as they
   * guarantee with high likelihood that all but very close points are put in separate clusters
   * initially. The clusters themselves are actually collapsed periodically when their number goes
   * over the maximum number of clusters and the distanceCutoff is increased.
   * So, the returned value is only an initial estimate.
   * @param data the datapoints whose distance is to be estimated.
   * @param distanceMeasure the distance measure used to compute the distance between two points.
   * @return the minimum distance between the first sampleLimit points
   * @see org.apache.mahout.clustering.streaming.cluster.StreamingKMeans#clusterInternal(Iterable, boolean)
   */
  public static double estimateDistanceCutoff(List<? extends Vector> data, DistanceMeasure distanceMeasure) {
    BruteSearch searcher = new BruteSearch(distanceMeasure);
    searcher.addAll(data);
    double minDistance = Double.POSITIVE_INFINITY;
    for (Vector vector : data) {
      double closest = searcher.searchFirst(vector, true).getWeight();
      if (minDistance > 0 && closest < minDistance) {
        minDistance = closest;
      }
      searcher.add(vector);
    }
    return minDistance;
  }

  public static double estimateDistanceCutoff(Iterable<? extends Vector> data, DistanceMeasure distanceMeasure,
                                              int sampleLimit) {
    return estimateDistanceCutoff(Lists.newArrayList(Iterables.limit(data, sampleLimit)), distanceMeasure);
  }

  /**
   * Computes the Davies-Bouldin Index for a given clustering.
   * See http://en.wikipedia.org/wiki/Clustering_algorithm#Internal_evaluation
   * @param centroids list of centroids
   * @param distanceMeasure distance measure for inter-cluster distances
   * @param clusterDistanceSummaries summaries of the clusters; See summarizeClusterDistances
   * @return the Davies-Bouldin Index
   */
  public static double daviesBouldinIndex(List<? extends Vector> centroids, DistanceMeasure distanceMeasure,
                                          List<OnlineSummarizer> clusterDistanceSummaries) {
    Preconditions.checkArgument(centroids.size() == clusterDistanceSummaries.size(),
        "Number of centroids and cluster summaries differ.");
    int n = centroids.size();
    double totalDBIndex = 0;
    // The inner loop shouldn't be reduced for j = i + 1 to n because the computation of the Davies-Bouldin
    // index is not really symmetric.
    // For a given cluster i, we look for a cluster j that maximizes the ratio of the sum of average distances
    // from points in cluster i to its center and and points in cluster j to its center to the distance between
    // cluster i and cluster j.
    // The maximization is the key issue, as the cluster that maximizes this ratio might be j for i but is NOT
    // NECESSARILY i for j.
    for (int i = 0; i < n; ++i) {
      double averageDistanceI = clusterDistanceSummaries.get(i).getMean();
      double maxDBIndex = 0;
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          double dbIndex = (averageDistanceI + clusterDistanceSummaries.get(j).getMean())
              / distanceMeasure.distance(centroids.get(i), centroids.get(j));
          if (dbIndex > maxDBIndex) {
            maxDBIndex = dbIndex;
          }
        }
      }
      totalDBIndex += maxDBIndex;
    }
    return totalDBIndex / n;
  }

  /**
   * Computes the Dunn Index of a given clustering. See http://en.wikipedia.org/wiki/Dunn_index
   * @param centroids list of centroids
   * @param distanceMeasure distance measure to compute inter-centroid distance with
   * @param clusterDistanceSummaries summaries of the clusters; See summarizeClusterDistances
   * @return the Dunn Index
   */
  public static double dunnIndex(List<? extends Vector> centroids, DistanceMeasure distanceMeasure,
                                 List<OnlineSummarizer> clusterDistanceSummaries) {
    Preconditions.checkArgument(centroids.size() == clusterDistanceSummaries.size(),
        "Number of centroids and cluster summaries differ.");
    int n = centroids.size();
    // Intra-cluster distances will come from the OnlineSummarizer, and will be the median distance (noting that
    // the median for just one value is that value).
    // A variety of metrics can be used for the intra-cluster distance including max distance between two points,
    // mean distance, etc. Median distance was chosen as this is more robust to outliers and characterizes the
    // distribution of distances (from a point to the center) better.
    double maxIntraClusterDistance = 0;
    for (OnlineSummarizer summarizer : clusterDistanceSummaries) {
      if (summarizer.getCount() > 0) {
        double intraClusterDistance;
        if (summarizer.getCount() == 1) {
          intraClusterDistance = summarizer.getMean();
        } else {
          intraClusterDistance = summarizer.getMedian();
        }
        if (maxIntraClusterDistance < intraClusterDistance) {
          maxIntraClusterDistance = intraClusterDistance;
        }
      }
    }
    double minDunnIndex = Double.POSITIVE_INFINITY;
    for (int i = 0; i < n; ++i) {
      // Distances are symmetric, so d(i, j) = d(j, i).
      for (int j = i + 1; j < n; ++j) {
        double dunnIndex = distanceMeasure.distance(centroids.get(i), centroids.get(j));
        if (minDunnIndex > dunnIndex) {
          minDunnIndex = dunnIndex;
        }
      }
    }
    return minDunnIndex / maxIntraClusterDistance;
  }

  public static double choose2(double n) {
    return n * (n - 1) / 2;
  }

  /**
   * Creates a confusion matrix by searching for the closest cluster of both the row clustering and column clustering
   * of a point and adding its weight to that cell of the matrix.
   * It doesn't matter which clustering is the row clustering and which is the column clustering. If they're
   * interchanged, the resulting matrix is the transpose of the original one.
   * @param rowCentroids clustering one
   * @param columnCentroids clustering two
   * @param datapoints datapoints whose closest cluster we need to find
   * @param distanceMeasure distance measure to use
   * @return the confusion matrix
   */
  public static Matrix getConfusionMatrix(List<? extends Vector> rowCentroids, List<? extends  Vector> columnCentroids,
                                          Iterable<? extends Vector> datapoints, DistanceMeasure distanceMeasure) {
    Searcher rowSearcher = new BruteSearch(distanceMeasure);
    rowSearcher.addAll(rowCentroids);
    Searcher columnSearcher = new BruteSearch(distanceMeasure);
    columnSearcher.addAll(columnCentroids);

    int numRows = rowCentroids.size();
    int numCols = columnCentroids.size();
    Matrix confusionMatrix = new DenseMatrix(numRows, numCols);

    for (Vector vector : datapoints) {
      WeightedThing<Vector> closestRowCentroid = rowSearcher.search(vector, 1).get(0);
      WeightedThing<Vector> closestColumnCentroid = columnSearcher.search(vector, 1).get(0);
      int row = ((Centroid) closestRowCentroid.getValue()).getIndex();
      int column = ((Centroid) closestColumnCentroid.getValue()).getIndex();
      double vectorWeight;
      if (vector instanceof WeightedVector) {
        vectorWeight = ((WeightedVector) vector).getWeight();
      } else {
        vectorWeight = 1;
      }
      confusionMatrix.set(row, column, confusionMatrix.get(row, column) + vectorWeight);
    }

    return confusionMatrix;
  }

  /**
   * Computes the Adjusted Rand Index for a given confusion matrix.
   * @param confusionMatrix confusion matrix; not to be confused with the more restrictive ConfusionMatrix class
   * @return the Adjusted Rand Index
   */
  public static double getAdjustedRandIndex(Matrix confusionMatrix) {
    int numRows = confusionMatrix.numRows();
    int numCols = confusionMatrix.numCols();
    double rowChoiceSum = 0;
    double columnChoiceSum = 0;
    double totalChoiceSum = 0;
    double total = 0;
    for (int i = 0; i < numRows; ++i) {
      double rowSum = 0;
      for (int j = 0; j < numCols; ++j) {
        rowSum += confusionMatrix.get(i, j);
        totalChoiceSum += choose2(confusionMatrix.get(i, j));
      }
      total += rowSum;
      rowChoiceSum += choose2(rowSum);
    }
    for (int j = 0; j < numCols; ++j) {
      double columnSum = 0;
      for (int i = 0; i < numRows; ++i) {
        columnSum += confusionMatrix.get(i, j);
      }
      columnChoiceSum += choose2(columnSum);
    }
    double rowColumnChoiceSumDivTotal = rowChoiceSum * columnChoiceSum / choose2(total);
    return (totalChoiceSum - rowColumnChoiceSumDivTotal)
        / ((rowChoiceSum + columnChoiceSum) / 2 - rowColumnChoiceSumDivTotal);
  }

  /**
   * Computes the total weight of the points in the given Vector iterable.
   * @param data iterable of points
   * @return total weight
   */
  public static double totalWeight(Iterable<? extends Vector> data) {
    double sum = 0;
    for (Vector row : data) {
      Preconditions.checkNotNull(row);
      if (row instanceof WeightedVector) {
        sum += ((WeightedVector)row).getWeight();
      } else {
        sum++;
      }
    }
    return sum;
  }
}
