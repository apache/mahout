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

package org.apache.mahout.clustering.cdbw;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.GaussianAccumulator;
import org.apache.mahout.clustering.OnlineGaussianAccumulator;
import org.apache.mahout.clustering.evaluation.RepresentativePointsDriver;
import org.apache.mahout.clustering.evaluation.RepresentativePointsMapper;
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
import com.google.common.collect.Maps;

/**
 * This class calculates the CDbw metric as defined in
 * http://www.db-net.aueb.gr/index.php/corporate/content/download/227/833/file/HV_poster2002.pdf
 */
public final class CDbwEvaluator {
  
  private static final Logger log = LoggerFactory.getLogger(CDbwEvaluator.class);
  
  private final Map<Integer,List<VectorWritable>> representativePoints;
  private final Map<Integer,Double> stDevs = Maps.newHashMap();
  private final List<Cluster> clusters;
  private final DistanceMeasure measure;
  private Double interClusterDensity = null;
  // these are symmetric so we only compute half of them
  private Map<Integer,Map<Integer,Double>> minimumDistances = null;
  // these are symmetric too
  private Map<Integer,Map<Integer,Double>> interClusterDensities = null;
  // these are symmetric too
  private Map<Integer,Map<Integer,int[]>> closestRepPointIndices = null;
  
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
  public CDbwEvaluator(Map<Integer,List<VectorWritable>> representativePoints, List<Cluster> clusters,
      DistanceMeasure measure) {
    this.representativePoints = representativePoints;
    this.clusters = clusters;
    this.measure = measure;
    for (Integer cId : representativePoints.keySet()) {
      computeStd(cId);
    }
  }
  
  /**
   * Initialize a new instance from job information
   * 
   * @param conf
   *          a Configuration with appropriate parameters
   * @param clustersIn
   *          a String path to the input clusters directory
   */
  public CDbwEvaluator(Configuration conf, Path clustersIn) {
    measure = ClassUtils
        .instantiateAs(conf.get(RepresentativePointsDriver.DISTANCE_MEASURE_KEY), DistanceMeasure.class);
    representativePoints = RepresentativePointsMapper.getRepresentativePoints(conf);
    clusters = loadClusters(conf, clustersIn);
    for (Integer cId : representativePoints.keySet()) {
      computeStd(cId);
    }
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
   * Compute the standard deviation of the representative points for the given cluster. Store these in stDevs, indexed
   * by cI
   * 
   * @param cI
   *          a int clusterId.
   */
  private void computeStd(int cI) {
    List<VectorWritable> repPts = representativePoints.get(cI);
    GaussianAccumulator accumulator = new OnlineGaussianAccumulator();
    for (VectorWritable vw : repPts) {
      accumulator.observe(vw.get(), 1.0);
    }
    accumulator.compute();
    double d = accumulator.getAverageStd();
    stDevs.put(cI, d);
  }
  
  /**
   * Compute the density of points near the midpoint between the two closest points of the clusters (eqn 2) used for
   * inter-cluster density calculation
   * 
   * @param uIJ
   *          the Vector midpoint between the closest representative points of the clusters
   * @param cI
   *          the int clusterId of the i-th cluster
   * @param cJ
   *          the int clusterId of the j-th cluster
   * @param avgStd
   *          the double average standard deviation of the two clusters
   * @return a double
   */
  private double density(Vector uIJ, int cI, int cJ, double avgStd) {
    List<VectorWritable> repI = representativePoints.get(cI);
    List<VectorWritable> repJ = representativePoints.get(cJ);
    double sum = 0.0;
    // count the number of representative points of the clusters which are within the
    // average std of the two clusters from the midpoint uIJ (eqn 3)
    for (VectorWritable vwI : repI) {
      if (uIJ != null && measure.distance(uIJ, vwI.get()) <= avgStd) {
        sum++;
      }
    }
    for (VectorWritable vwJ : repJ) {
      if (uIJ != null && measure.distance(uIJ, vwJ.get()) <= avgStd) {
        sum++;
      }
    }
    int nI = repI.size();
    int nJ = repJ.size();
    return sum / (nI + nJ);
  }
  
  /**
   * Compute the CDbw validity metric (eqn 8). The goal of this metric is to reward clusterings which have a high
   * intraClusterDensity and also a high cluster separation.
   * 
   * @return a double
   */
  public double getCDbw() {
    return intraClusterDensity() * separation();
  }
  
  /**
   * The average density within clusters is defined as the percentage of representative points that reside in the
   * neighborhood of the clusters' centers. The goal is the density within clusters to be significantly high. (eqn 5)
   * 
   * @return a double
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
    return avgDensity / count;
  }
  
  /**
   * This function evaluates the density of points in the regions between each clusters (eqn 1). The goal is the density
   * in the area between clusters to be significant low.
   * 
   * @return a Map<Integer,Map<Integer,Double>> of the inter-cluster densities
   */
  public Map<Integer,Map<Integer,Double>> interClusterDensities() {
    if (interClusterDensities != null) {
      return interClusterDensities;
    }
    interClusterDensities = new TreeMap<Integer,Map<Integer,Double>>();
    // find the closest representative points between the clusters
    for (int i = 0; i < clusters.size(); i++) {
      int cI = clusters.get(i).getId();
      Map<Integer,Double> map = new TreeMap<Integer,Double>();
      interClusterDensities.put(cI, map);
      for (int j = i + 1; j < clusters.size(); j++) {
        int cJ = clusters.get(j).getId();
        double minDistance = minimumDistance(cI, cJ); // the distance between the closest representative points
        Vector uIJ = midpointVector(cI, cJ); // the midpoint between the closest representative points
        double stdSum = stDevs.get(cI) + stDevs.get(cJ);
        double density = density(uIJ, cI, cJ, stdSum / 2);
        double interDensity = minDistance * density / stdSum;
        map.put(cJ, interDensity);
        if (log.isDebugEnabled()) {
          log.debug("minDistance[{},{}]={}", cI, cJ, minDistance);
          log.debug("interDensity[{},{}]={}", cI, cJ, density);
          log.debug("density[{},{}]={}", cI, cJ, interDensity);
        }
      }
    }
    return interClusterDensities;
  }
  
  /**
   * Calculate the separation of clusters (eqn 4) taking into account both the distances between the clusters' closest
   * points and the Inter-cluster density. The goal is the distances between clusters to be high while the
   * representative point density in the areas between them are low.
   * 
   * @return a double
   */
  public double separation() {
    double minDistanceSum = 0;
    Map<Integer,Map<Integer,Double>> distances = minimumDistances();
    for (Map<Integer,Double> map : distances.values()) {
      for (Double dist : map.values()) {
        if (!Double.isInfinite(dist)) {
          minDistanceSum += dist * 2; // account for other half of calculated triangular minimumDistances matrix
        }
      }
    }
    return minDistanceSum / (1.0 + interClusterDensity());
  }
  
  /**
   * This function evaluates the average density of points in the regions between clusters (eqn 1). The goal is the
   * density in the area between clusters to be significant low.
   * 
   * @return a double
   */
  public double interClusterDensity() {
    if (interClusterDensity != null) {
      return interClusterDensity;
    }
    double sum = 0.0;
    int count = 0;
    Map<Integer,Map<Integer,Double>> distances = interClusterDensities();
    for (Map<Integer,Double> row : distances.values()) {
      for (Double density : row.values()) {
        if (!Double.isNaN(density)) {
          sum += density;
          count++;
        }
      }
    }
    log.debug("interClusterDensity={}", sum);
    interClusterDensity = sum / count;
    return interClusterDensity;
  }
  
  /**
   * The average density within clusters is defined as the percentage of representative points that reside in the
   * neighborhood of the clusters' centers. The goal is the density within clusters to be significantly high. (eqn 5)
   * 
   * @return a Vector of the intra-densities of each clusterId
   */
  public Vector intraClusterDensities() {
    Vector densities = new RandomAccessSparseVector(Integer.MAX_VALUE);
    // compute the average standard deviation of the clusters
    double stdev = 0.0;
    for (Integer cI : representativePoints.keySet()) {
      stdev += stDevs.get(cI);
    }
    int c = representativePoints.size();
    stdev /= c;
    for (Cluster cluster : clusters) {
      Integer cI = cluster.getId();
      List<VectorWritable> repPtsI = representativePoints.get(cI);
      int r = repPtsI.size();
      double sumJ = 0.0;
      // compute the term density (eqn 6)
      for (VectorWritable pt : repPtsI) {
        // compute f(x, vIJ) (eqn 7)
        Vector repJ = pt.get();
        double densityIJ = measure.distance(cluster.getCenter(), repJ) <= stdev ? 1.0 : 0.0;
        // accumulate sumJ
        sumJ += densityIJ / stdev;
      }
      densities.set(cI, sumJ / r);
    }
    return densities;
  }
  
  /**
   * Calculate and cache the distances between the clusters' closest representative points. Also cache the indices of
   * the closest representative points used for later use
   * 
   * @return a Map<Integer,Vector> of the closest distances, keyed by clusterId
   */
  private Map<Integer,Map<Integer,Double>> minimumDistances() {
    if (minimumDistances != null) {
      return minimumDistances;
    }
    minimumDistances = new TreeMap<Integer,Map<Integer,Double>>();
    closestRepPointIndices = new TreeMap<Integer,Map<Integer,int[]>>();
    for (int i = 0; i < clusters.size(); i++) {
      Integer cI = clusters.get(i).getId();
      Map<Integer,Double> map = new TreeMap<Integer,Double>();
      Map<Integer,int[]> treeMap = new TreeMap<Integer,int[]>();
      closestRepPointIndices.put(cI, treeMap);
      minimumDistances.put(cI, map);
      List<VectorWritable> closRepI = representativePoints.get(cI);
      for (int j = i + 1; j < clusters.size(); j++) {
        // find min{d(closRepI, closRepJ)}
        Integer cJ = clusters.get(j).getId();
        List<VectorWritable> closRepJ = representativePoints.get(cJ);
        double minDistance = Double.MAX_VALUE;
        int[] midPointIndices = null;
        for (int xI = 0; xI < closRepI.size(); xI++) {
          VectorWritable aRepI = closRepI.get(xI);
          for (int xJ = 0; xJ < closRepJ.size(); xJ++) {
            VectorWritable aRepJ = closRepJ.get(xJ);
            double distance = measure.distance(aRepI.get(), aRepJ.get());
            if (distance < minDistance) {
              minDistance = distance;
              midPointIndices = new int[] {xI, xJ};
            }
          }
        }
        map.put(cJ, minDistance);
        treeMap.put(cJ, midPointIndices);
      }
    }
    return minimumDistances;
  }
  
  private double minimumDistance(int cI, int cJ) {
    Map<Integer,Double> distances = minimumDistances().get(cI);
    if (distances != null) {
      return distances.get(cJ);
    } else {
      return minimumDistances().get(cJ).get(cI);
    }
  }
  
  private Vector midpointVector(int cI, int cJ) {
    Map<Integer,Double> distances = minimumDistances().get(cI);
    if (distances != null) {
      int[] ks = closestRepPointIndices.get(cI).get(cJ);
      if (ks == null) {
        return null;
      }
      return representativePoints.get(cI).get(ks[0]).get().plus(representativePoints.get(cJ).get(ks[1]).get())
          .divide(2);
    } else {
      int[] ks = closestRepPointIndices.get(cJ).get(cI);
      if (ks == null) {
        return null;
      }
      return representativePoints.get(cJ).get(ks[1]).get().plus(representativePoints.get(cI).get(ks[0]).get())
          .divide(2);
    }
    
  }
}
