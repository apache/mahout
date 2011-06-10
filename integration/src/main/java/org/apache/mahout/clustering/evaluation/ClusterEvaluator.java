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

package org.apache.mahout.clustering.evaluation;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ClusterEvaluator {

  private static final Logger log = LoggerFactory.getLogger(ClusterEvaluator.class);

  private final Map<Integer, List<VectorWritable>> representativePoints;

  private final List<Cluster> clusters;

  private final DistanceMeasure measure;

  private boolean pruned;

  /**
   * For testing only
   * 
   * @param representativePoints
   *            a Map<Integer,List<VectorWritable>> of representative points keyed by clusterId
   * @param clusters
   *            a Map<Integer,Cluster> of the clusters keyed by clusterId
   * @param measure
   *            an appropriate DistanceMeasure
   */
  public ClusterEvaluator(Map<Integer, List<VectorWritable>> representativePoints,
                          List<Cluster> clusters, DistanceMeasure measure) {
    this.representativePoints = representativePoints;
    this.clusters = clusters;
    this.measure = measure;
  }

  /**
   * Initialize a new instance from job information
   * 
   * @param conf
   *            a Configuration with appropriate parameters
   * @param clustersIn
   *            a String path to the input clusters directory
   */
  public ClusterEvaluator(Configuration conf, Path clustersIn)
    throws ClassNotFoundException, InstantiationException, IllegalAccessException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    measure = ccl.loadClass(conf.get(RepresentativePointsDriver.DISTANCE_MEASURE_KEY)).asSubclass(DistanceMeasure.class)
        .newInstance();
    representativePoints = RepresentativePointsMapper.getRepresentativePoints(conf);
    clusters = loadClusters(conf, clustersIn);
  }

  /**
   * Load the clusters from their sequence files
   * 
   * @param clustersIn 
   *            a String pathname to the directory containing input cluster files
   * @return a List<Cluster> of the clusters
   */
  private static List<Cluster> loadClusters(Configuration conf, Path clustersIn) {
    List<Cluster> clusters = Lists.newArrayList();
    for (Cluster value :
         new SequenceFileDirValueIterable<Cluster>(clustersIn, PathType.LIST, PathFilters.logsCRCFilter(), conf)) {
      clusters.add(value);
    }
    return clusters;
  }

  /**
   * Return if the cluster is valid. Valid clusters must have more than 2 representative points,
   * and at least one of them must be different than the cluster center. This is because the
   * representative points extraction will duplicate the cluster center if it is empty.
   * 
   * @param clusterI a Cluster
   * @return a boolean
   */
  private boolean invalidCluster(Cluster clusterI) {
    List<VectorWritable> repPts = representativePoints.get(clusterI.getId());
    if (repPts.size() < 2) {
      return true;
    }
    for (VectorWritable vw : repPts) {
      Vector vector = vw.get();
      if (!vector.equals(clusterI.getCenter())) {
        return false;
      }
    }
    return true;
  }

  private void pruneInvalidClusters() {
    if (pruned) {
      return;
    }
    for (Iterator<Cluster> it = clusters.iterator(); it.hasNext();) {
      Cluster cluster = it.next();
      if (invalidCluster(cluster)) {
        log.info("Pruning cluster Id=" + cluster.getId());
        it.remove();
        representativePoints.remove(cluster.getId());
      }
    }
    pruned = true;
  }

  /**
   * Computes the inter-cluster density as defined in "Mahout In Action"
   * 
   * @return the interClusterDensity
   */
  public double interClusterDensity() {
    pruneInvalidClusters();
    double max = 0;
    double min = Double.MAX_VALUE;
    double sum = 0;
    int count = 0;
    for (int i = 0; i < clusters.size(); i++) {
      Cluster clusterI = clusters.get(i);
      for (int j = i + 1; j < clusters.size(); j++) {
        Cluster clusterJ = clusters.get(j);
        double d = measure.distance(clusterI.getCenter(), clusterJ.getCenter());
        min = Math.min(d, min);
        max = Math.max(d, max);
        sum += d;
        count++;
      }
    }
    double density = (sum / count - min) / (max - min);
    log.info("Inter-Cluster Density = " + density);
    return density;
  }

  /**
   * Computes the intra-cluster density as the average distance of the representative points
   * from each other
   * 
   * @return the intraClusterDensity of the representativePoints
   */
  public double intraClusterDensity() {
    pruneInvalidClusters();
    double avgDensity = 0;
    for (Cluster cluster : clusters) {
      int count = 0;
      double max = 0;
      double min = Double.MAX_VALUE;
      double sum = 0;
      List<VectorWritable> repPoints = representativePoints.get(cluster.getId());
      for (int i = 0; i < repPoints.size(); i++) {
        for (int j = i + 1; j < repPoints.size(); j++) {
          double d = measure.distance(repPoints.get(i).get(), repPoints.get(j).get());
          min = Math.min(d, min);
          max = Math.max(d, max);
          sum += d;
          count++;
        }
      }
      double density = (sum / count - min) / (max - min);
      avgDensity += density;
      log.info("Intra-Cluster Density[" + cluster.getId() + "] = " + density);
    }
    avgDensity = clusters.isEmpty() ? 0 : avgDensity / clusters.size();
    log.info("Intra-Cluster Density = " + avgDensity);
    return avgDensity;

  }
}
