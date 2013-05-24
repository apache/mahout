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

import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

public class ClusterEvaluator {
  
  private static final Logger log = LoggerFactory.getLogger(ClusterEvaluator.class);
  
  private final Map<Integer,List<VectorWritable>> representativePoints;
  
  private final List<Cluster> clusters;
  
  private final DistanceMeasure measure;
  
  /**
   * For testing only
   * 
   * @param representativePoints
   *          a Map<Integer,List<VectorWritable>> of representative points keyed by clusterId
   * @param clusters
   *          a Map<Integer,Cluster> of the clusters keyed by clusterId
   * @param measure
   *          an appropriate DistanceMeasure
   */
  public ClusterEvaluator(Map<Integer,List<VectorWritable>> representativePoints, List<Cluster> clusters,
      DistanceMeasure measure) {
    this.representativePoints = representativePoints;
    this.clusters = clusters;
    this.measure = measure;
  }
  
  /**
   * Initialize a new instance from job information
   * 
   * @param conf
   *          a Configuration with appropriate parameters
   * @param clustersIn
   *          a String path to the input clusters directory
   */
  public ClusterEvaluator(Configuration conf, Path clustersIn) {
    measure = ClassUtils
        .instantiateAs(conf.get(RepresentativePointsDriver.DISTANCE_MEASURE_KEY), DistanceMeasure.class);
    representativePoints = RepresentativePointsMapper.getRepresentativePoints(conf);
    clusters = loadClusters(conf, clustersIn);
  }
  
  /**
   * Load the clusters from their sequence files
   * 
   * @param clustersIn
   *          a String pathname to the directory containing input cluster files
   * @return a List<Cluster> of the clusters
   */
  private static List<Cluster> loadClusters(Configuration conf, Path clustersIn) {
    List<Cluster> clusters = Lists.newArrayList();
    for (ClusterWritable clusterWritable : new SequenceFileDirValueIterable<ClusterWritable>(clustersIn, PathType.LIST,
        PathFilters.logsCRCFilter(), conf)) {
      Cluster cluster = clusterWritable.getValue();
      clusters.add(cluster);
    }
    return clusters;
  }
  
  /**
   * Computes the inter-cluster density as defined in "Mahout In Action"
   * 
   * @return the interClusterDensity
   */
  public double interClusterDensity() {
    double max = Double.NEGATIVE_INFINITY;
    double min = Double.POSITIVE_INFINITY;
    double sum = 0;
    int count = 0;
    Map<Integer,Vector> distances = interClusterDistances();
    for (Vector row : distances.values()) {
      for (Element element : row.nonZeroes()) {
        double d = element.get();
        min = Math.min(d, min);
        max = Math.max(d, max);
        sum += d;
        count++;
      }
    }
    double density = (sum / count - min) / (max - min);
    log.info("Scaled Inter-Cluster Density = {}", density);
    return density;
  }
  
  /**
   * Computes the inter-cluster distances
   * 
   * @return a Map<Integer, Vector>
   */
  public Map<Integer,Vector> interClusterDistances() {
    Map<Integer,Vector> distances = new TreeMap<Integer,Vector>();
    for (int i = 0; i < clusters.size(); i++) {
      Cluster clusterI = clusters.get(i);
      RandomAccessSparseVector row = new RandomAccessSparseVector(Integer.MAX_VALUE);
      distances.put(clusterI.getId(), row);
      for (int j = i + 1; j < clusters.size(); j++) {
        Cluster clusterJ = clusters.get(j);
        double d = measure.distance(clusterI.getCenter(), clusterJ.getCenter());
        row.set(clusterJ.getId(), d);
      }
    }
    return distances;
  }
  
  /**
   * Computes the average intra-cluster density as the average of each cluster's intra-cluster density
   * 
   * @return the average intraClusterDensity
   */
  public double intraClusterDensity() {
    double avgDensity = 0;
    int count = 0;
    for (Element elem : intraClusterDensities().nonZeroes()) {
      double value = elem.get();
      if (!Double.isNaN(value)) {
        avgDensity += value;
        count++;
      }
    }
    avgDensity = clusters.isEmpty() ? 0 : avgDensity / count;
    log.info("Average Intra-Cluster Density = {}", avgDensity);
    return avgDensity;
  }
  
  /**
   * Computes the intra-cluster densities for all clusters as the average distance of the representative points from
   * each other
   * 
   * @return a Vector of the intraClusterDensity of the representativePoints by clusterId
   */
  public Vector intraClusterDensities() {
    Vector densities = new RandomAccessSparseVector(Integer.MAX_VALUE);
    for (Cluster cluster : clusters) {
      int count = 0;
      double max = Double.NEGATIVE_INFINITY;
      double min = Double.POSITIVE_INFINITY;
      double sum = 0;
      List<VectorWritable> repPoints = representativePoints.get(cluster.getId());
      for (int i = 0; i < repPoints.size(); i++) {
        for (int j = i + 1; j < repPoints.size(); j++) {
          Vector v1 = repPoints.get(i).get();
          Vector v2 = repPoints.get(j).get();
          double d = measure.distance(v1, v2);
          min = Math.min(d, min);
          max = Math.max(d, max);
          sum += d;
          count++;
        }
      }
      double density = (sum / count - min) / (max - min);
      densities.set(cluster.getId(), density);
      log.info("Intra-Cluster Density[{}] = {}", cluster.getId(), density);
    }
    return densities;
  }
}
