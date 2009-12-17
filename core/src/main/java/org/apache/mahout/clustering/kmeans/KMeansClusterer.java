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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class implements the k-means clustering algorithm. It uses
 * {@link Cluster} as a cluster representation. The class can be used as part of
 * a clustering job to be started as map/reduce job.
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
   * Iterates over all clusters and identifies the one closes to the given
   * point. Distance measure used is configured at creation time of
   * {@link KMeansClusterer}.
   * 
   * @param point
   *          a point to find a cluster for.
   * @param clusters
   *          a List<Cluster> to test.
   */
  public void emitPointToNearestCluster(Vector point,
      List<Cluster> clusters, OutputCollector<Text, KMeansInfo> output) throws IOException {
    Cluster nearestCluster = null;
    double nearestDistance = Double.MAX_VALUE;
    for (Cluster cluster : clusters) {
      Vector clusterCenter = cluster.getCenter();
      double distance = this.measure.distance(clusterCenter.getLengthSquared(),
          clusterCenter, point);
      log.info("{} Cluster: {}", distance, cluster.getId());
      if (distance < nearestDistance || nearestCluster == null) {
        nearestCluster = cluster;
        nearestDistance = distance;
      }
    }
    // emit only clusterID
    output.collect(new Text(nearestCluster.getIdentifier()), new KMeansInfo(1, point));
  }

  public void outputPointWithClusterInfo(Vector point,
      List<Cluster> clusters, OutputCollector<Text, Text> output) throws IOException {
    Cluster nearestCluster = null;
    double nearestDistance = Double.MAX_VALUE;
    for (Cluster cluster : clusters) {
      Vector clusterCenter = cluster.getCenter();
      double distance = measure.distance(clusterCenter.getLengthSquared(),
          clusterCenter, point);
      if (distance < nearestDistance || nearestCluster == null) {
        nearestCluster = cluster;
        nearestDistance = distance;
      }
    }
    
    String name = point.getName();
    String key = name != null && name.length() != 0 ? name : point.asFormatString();
    output.collect(new Text(key), new Text(String.valueOf(nearestCluster.getId())));
  }
}
