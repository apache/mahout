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

import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.GaussianAccumulator;
import org.apache.mahout.clustering.OnlineGaussianAccumulator;
import org.apache.mahout.clustering.evaluation.RepresentativePointsDriver;
import org.apache.mahout.clustering.evaluation.RepresentativePointsMapper;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class calculates the CDbw metric as defined in
 * http://www.db-net.aueb.gr/index.php/corporate/content/download/227/833/file/HV_poster2002.pdf 
 */
public class CDbwEvaluator {

  private static final Logger log = LoggerFactory.getLogger(CDbwEvaluator.class);

  private final Map<Integer, List<VectorWritable>> representativePoints;
  private final Map<Integer, Double> stDevs = Maps.newHashMap();
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
  public CDbwEvaluator(Map<Integer, List<VectorWritable>> representativePoints,
                       List<Cluster> clusters,
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
   *            a Configuration with appropriate parameters
   * @param clustersIn
   *            a String path to the input clusters directory
   */
  public CDbwEvaluator(Configuration conf, Path clustersIn) {
    measure = ClassUtils.instantiateAs(conf.get(RepresentativePointsDriver.DISTANCE_MEASURE_KEY),
                                       DistanceMeasure.class);
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
   * Compute the standard deviation of the representative points for the given cluster.
   * Store these in stDevs, indexed by cI
   * 
   * @param cI a int clusterId. 
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
        log.info("Pruning cluster Id={}", cluster.getId());
        it.remove();
        representativePoints.remove(cluster.getId());
      }
    }
    pruned = true;
  }

  /**
   * Compute the term density (eqn 2) used for inter-cluster density calculation
   * 
   * @param uIJ the Vector midpoint between the closest representative of the clusters
   * @param cI the int clusterId of the i-th cluster
   * @param cJ the int clusterId of the j-th cluster
   * @return a double
   */
  double interDensity(Vector uIJ, int cI, int cJ) {
    List<VectorWritable> repI = representativePoints.get(cI);
    List<VectorWritable> repJ = representativePoints.get(cJ);
    double sum = 0.0;
    Double stdevI = stDevs.get(cI);
    Double stdevJ = stDevs.get(cJ);
    // count the number of representative points of the clusters which are within the
    // average std of the two clusters from the midpoint uIJ (eqn 3)
    double avgStd = (stdevI + stdevJ) / 2.0;
    for (VectorWritable vwI : repI) {
      if (measure.distance(uIJ, vwI.get()) <= avgStd) {
        sum++;
      }
    }
    for (VectorWritable vwJ : repJ) {
      if (measure.distance(uIJ, vwJ.get()) <= avgStd) {
        sum++;
      }
    }
    int nI = repI.size();
    int nJ = repJ.size();
    return sum / (nI + nJ);
  }

  /**
   * Compute the CDbw validity metric (eqn 8). The goal of this metric is to reward clusterings which
   * have a high intraClusterDensity and also a high cluster separation.
   * 
   * @return a double
   */
  public double getCDbw() {
    pruneInvalidClusters();
    return intraClusterDensity() * separation();
  }

  /**
   * The average density within clusters is defined as the percentage of representative points that reside 
   * in the neighborhood of the clusters' centers. The goal is the density within clusters to be 
   * significantly high. (eqn 5)
   * 
   * @return a double
   */
  public double intraClusterDensity() {
    pruneInvalidClusters();
    // compute the average standard deviation of the clusters
    double stdev = 0.0;
    for (Integer cI : representativePoints.keySet()) {
      stdev += stDevs.get(cI);
    }
    int c = representativePoints.size();
    stdev /= c;
    // accumulate the summations
    double sumI = 0.0;
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
      // accumulate sumI
      sumI += sumJ / r;
    }
    return sumI / c;
  }

  /**
   * Calculate the separation of clusters (eqn 4) taking into account both the distances between the
   * clusters' closest points and the Inter-cluster density. The goal is the distances between clusters 
   * to be high while the representative point density in the areas between them are low.
   * 
   * @return a double
   */
  public double separation() {
    pruneInvalidClusters();
    double minDistanceSum = 0;
    for (int i = 0; i < clusters.size(); i++) {
      Integer cI = clusters.get(i).getId();
      List<VectorWritable> closRepI = representativePoints.get(cI);
      for (int j = 0; j < clusters.size(); j++) {
        if (i == j) {
          continue;
        }
        // find min{d(closRepI, closRepJ)}
        Integer cJ = clusters.get(j).getId();
        List<VectorWritable> closRepJ = representativePoints.get(cJ);
        double minDistance = Double.MAX_VALUE;
        for (VectorWritable aRepI : closRepI) {
          for (VectorWritable aRepJ : closRepJ) {
            double distance = measure.distance(aRepI.get(), aRepJ.get());
            if (distance < minDistance) {
              minDistance = distance;
            }
          }
        }
        minDistanceSum += minDistance;
      }
    }
    return minDistanceSum / (1.0 + interClusterDensity());
  }

  /**
   * This function evaluates the average density of points in the regions between clusters (eqn 1). 
   * The goal is the density in the area between clusters to be significant low.
   * 
   * @return a double
   */
  public double interClusterDensity() {
    pruneInvalidClusters();
    double sum = 0.0;
    // find the closest representative points between the clusters
    for (int i = 0; i < clusters.size(); i++) {
      Integer cI = clusters.get(i).getId();
      List<VectorWritable> repI = representativePoints.get(cI);
      for (int j = 1; j < clusters.size(); j++) {
        Integer cJ = clusters.get(j).getId();
        if (i == j) {
          continue;
        }
        List<VectorWritable> repJ = representativePoints.get(cJ);
        double minDistance = Double.MAX_VALUE; // the distance between the closest representative points
        Vector uIJ = null; // the midpoint between the closest representative points
        // find the closest representative points between the i-th and j-th clusters
        for (VectorWritable aRepI : repI) {
          for (VectorWritable aRepJ : repJ) {
            Vector closRepI = aRepI.get();
            Vector closRepJ = aRepJ.get();
            double distance = measure.distance(closRepI, closRepJ);
            if (distance < minDistance) {
              // set the distance and compute the midpoint
              minDistance = distance;
              uIJ = closRepI.plus(closRepJ).divide(2);
            }
          }
        }
        double stDevI = stDevs.get(cI);
        double stDevJ = stDevs.get(cJ);
        double interDensity = interDensity(uIJ, cI, cJ);
        double stdSum = stDevI + stDevJ;
        double density = 0.0;
        if (stdSum > 0.0) {
          density = minDistance * interDensity / stdSum;
        }

        log.debug("minDistance[{},{}]={}", new Object[] {cI, cJ, minDistance});
        log.debug("stDev[{}]={}", cI, stDevI);
        log.debug("stDev[{}]={}", cJ, stDevJ);
        log.debug("interDensity[{},{}]={}", new Object[] {cI, cJ, interDensity});
        log.debug("density[{},{}]={}", new Object[] {cI, cJ, density});

        sum += density;
      }
    }
    log.debug("interClusterDensity={}", sum);
    return sum;
  }
}
