/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.kmeans;

import java.io.IOException;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.ClusterObservations;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class implements the k-means clustering algorithm. It uses {@link Cluster} as a cluster
 * representation. The class can be used as part of a clustering job to be started as map/reduce job.
 * */
public class KMeansClusterer {

  private static final Logger log = LoggerFactory.getLogger(KMeansClusterer.class);

  /** Distance to use for point to cluster comparison. */
  private final DistanceMeasure measure;

  /**
   * Init the k-means clusterer with the distance measure to use for comparison.
   * 
   * @param measure
   *          The distance measure to use for comparing clusters against points.
   * 
   */
  public KMeansClusterer(DistanceMeasure measure) {
    this.measure = measure;
  }

  /**
   * Iterates over all clusters and identifies the one closes to the given point. Distance measure used is
   * configured at creation time.
   * 
   * @param point
   *          a point to find a cluster for.
   * @param clusters
   *          a List<Cluster> to test.
   */
  public void emitPointToNearestCluster(Vector point,
                                        Iterable<Cluster> clusters,
                                        Mapper<?,?,Text,ClusterObservations>.Context context)
    throws IOException, InterruptedException {
    Cluster nearestCluster = null;
    double nearestDistance = Double.MAX_VALUE;
    for (Cluster cluster : clusters) {
      Vector clusterCenter = cluster.getCenter();
      double distance = this.measure.distance(clusterCenter.getLengthSquared(), clusterCenter, point);
      if (KMeansClusterer.log.isDebugEnabled()) {
        log.debug("{} Cluster: {}", distance, cluster.getId());
      }
      if (distance < nearestDistance || nearestCluster == null) {
        nearestCluster = cluster;
        nearestDistance = distance;
      }
    }
    context.write(new Text(nearestCluster.getIdentifier()), new ClusterObservations(1, point, point.times(point)));
  }

  /**
   * Sequential implementation to add point to the nearest cluster
   * @param point
   * @param clusters
   */
  protected void addPointToNearestCluster(Vector point, Iterable<Cluster> clusters) {
    Cluster closestCluster = null;
    double closestDistance = Double.MAX_VALUE;
    for (Cluster cluster : clusters) {
      double distance = measure.distance(cluster.getCenter(), point);
      if (closestCluster == null || closestDistance > distance) {
        closestCluster = cluster;
        closestDistance = distance;
      }
    }
    closestCluster.observe(point, 1);
  }

  /**
   * Sequential implementation to test convergence and update cluster centers
   */
  protected boolean testConvergence(Iterable<Cluster> clusters, double distanceThreshold) {
    boolean converged = true;
    for (Cluster cluster : clusters) {
      if (!computeConvergence(cluster, distanceThreshold)) {
        converged = false;
      }
      cluster.computeParameters();
    }
    return converged;
  }

  public void outputPointWithClusterInfo(Vector vector,
                                         Iterable<Cluster> clusters,
                                         Mapper<?,?,IntWritable,WeightedVectorWritable>.Context context)
    throws IOException, InterruptedException {
    AbstractCluster nearestCluster = null;
    double nearestDistance = Double.MAX_VALUE;
    for (AbstractCluster cluster : clusters) {
      Vector clusterCenter = cluster.getCenter();
      double distance = measure.distance(clusterCenter.getLengthSquared(), clusterCenter, vector);
      if (distance < nearestDistance || nearestCluster == null) {
        nearestCluster = cluster;
        nearestDistance = distance;
      }
    }
    context.write(new IntWritable(nearestCluster.getId()), new WeightedVectorWritable(1, vector));
  }

  /**
   * Iterates over all clusters and identifies the one closes to the given point. Distance measure used is
   * configured at creation time.
   * 
   * @param point
   *          a point to find a cluster for.
   * @param clusters
   *          a List<Cluster> to test.
   */
  protected void emitPointToNearestCluster(Vector point, Iterable<Cluster> clusters, Writer writer)
    throws IOException {
    AbstractCluster nearestCluster = null;
    double nearestDistance = Double.MAX_VALUE;
    for (AbstractCluster cluster : clusters) {
      Vector clusterCenter = cluster.getCenter();
      double distance = this.measure.distance(clusterCenter.getLengthSquared(), clusterCenter, point);
      if (log.isDebugEnabled()) {
        log.debug("{} Cluster: {}", distance, cluster.getId());
      }
      if (distance < nearestDistance || nearestCluster == null) {
        nearestCluster = cluster;
        nearestDistance = distance;
      }
    }
    writer.append(new IntWritable(nearestCluster.getId()), new WeightedVectorWritable(1, point));
  }

  /**
   * This is the reference k-means implementation. Given its inputs it iterates over the points and clusters
   * until their centers converge or until the maximum number of iterations is exceeded.
   * 
   * @param points
   *          the input List<Vector> of points
   * @param clusters
   *          the List<Cluster> of initial clusters
   * @param measure
   *          the DistanceMeasure to use
   * @param maxIter
   *          the maximum number of iterations
   */
  public static List<List<Cluster>> clusterPoints(Iterable<Vector> points,
                                                  List<Cluster> clusters,
                                                  DistanceMeasure measure,
                                                  int maxIter,
                                                  double distanceThreshold) {
    List<List<Cluster>> clustersList = Lists.newArrayList();
    clustersList.add(clusters);

    boolean converged = false;
    int iteration = 0;
    while (!converged && iteration < maxIter) {
      log.info("Reference Iteration: " + iteration);
      List<Cluster> next = Lists.newArrayList();
      for (Cluster c : clustersList.get(iteration)) {
        next.add(new Cluster(c.getCenter(), c.getId(), measure));
      }
      clustersList.add(next);
      converged = runKMeansIteration(points, next, measure, distanceThreshold);
      iteration++;
    }
    return clustersList;
  }

  /**
   * Perform a single iteration over the points and clusters, assigning points to clusters and returning if
   * the iterations are completed.
   * 
   * @param points
   *          the List<Vector> having the input points
   * @param clusters
   *          the List<Cluster> clusters
   * @param measure
   *          a DistanceMeasure to use
   */
  protected static boolean runKMeansIteration(Iterable<Vector> points,
                                              Iterable<Cluster> clusters,
                                              DistanceMeasure measure,
                                              double distanceThreshold) {
    // iterate through all points, assigning each to the nearest cluster
    KMeansClusterer clusterer = new KMeansClusterer(measure);
    for (Vector point : points) {
      clusterer.addPointToNearestCluster(point, clusters);
    }
    return clusterer.testConvergence(clusters, distanceThreshold);
  }

  public boolean computeConvergence(Cluster cluster, double distanceThreshold) {
    return cluster.computeConvergence(measure, distanceThreshold);
  }

}
