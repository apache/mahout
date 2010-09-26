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

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.evaluation.RepresentativePointsDriver;
import org.apache.mahout.clustering.evaluation.RepresentativePointsMapper;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.SquareRootFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CDbwEvaluator {

  private static final Logger log = LoggerFactory.getLogger(CDbwEvaluator.class);

  private final Map<Integer, List<VectorWritable>> representativePoints;

  private final Map<Integer, Double> stDevs = new HashMap<Integer, Double>();

  private final Map<Integer, Cluster> clusters;

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
                       Map<Integer, Cluster> clusters,
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
   *            a JobConf with appropriate parameters
   * @param clustersIn
   *            a String path to the input clusters directory
   */
  public CDbwEvaluator(Configuration conf, Path clustersIn) throws ClassNotFoundException, InstantiationException,
      IllegalAccessException, IOException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    measure = ccl.loadClass(conf.get(RepresentativePointsDriver.DISTANCE_MEASURE_KEY)).asSubclass(DistanceMeasure.class)
        .newInstance();
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
  private static Map<Integer, Cluster> loadClusters(Configuration conf, Path clustersIn) throws InstantiationException,
      IllegalAccessException, IOException {
    Map<Integer, Cluster> clusters = new HashMap<Integer, Cluster>();
    FileSystem fs = clustersIn.getFileSystem(conf);
    for (FileStatus part : fs.listStatus(clustersIn)) {
      if (!part.getPath().getName().startsWith(".")) {
        Path inPart = part.getPath();
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, inPart, conf);
        Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
        Writable value = reader.getValueClass().asSubclass(Writable.class).newInstance();
        while (reader.next(key, value)) {
          Cluster cluster = (Cluster) value;
          clusters.put(cluster.getId(), cluster);
          value = reader.getValueClass().asSubclass(Writable.class).newInstance();
        }
        reader.close();
      }
    }
    return clusters;
  }

  private void computeStd(int cI) {
    List<VectorWritable> repPts = representativePoints.get(cI);
    int s0 = 0;
    Vector s1 = null;
    Vector s2 = null;
    for (VectorWritable vw : repPts) {
      s0++;
      Vector v = vw.get();
      s1 = s1 == null ? v.clone() : s1.plus(v);
      s2 = s2 == null ? v.times(v) : s2.plus(v.times(v));
    }
    if (s0 > 1) {
      Vector std = s2.times(s0).minus(s1.times(s1)).assign(new SquareRootFunction()).divide(s0);
      double d = std.zSum() / std.size();
      log.debug("stDev[" + cI + "]=" + d);
      stDevs.put(cI, d);
    }
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
    for (Iterator<Cluster> it = clusters.values().iterator(); it.hasNext();) {
      Cluster cluster = it.next();
      if (invalidCluster(cluster)) {
        log.info("Pruning cluster Id=" + cluster.getId());
        it.remove();
        representativePoints.remove(cluster.getId());
      }
    }
    pruned = true;
  }

  double interDensity(Vector uIJ, int cI, int cJ) {
    List<VectorWritable> repI = representativePoints.get(cI);
    List<VectorWritable> repJ = representativePoints.get(cJ);
    double density = 0.0;
    double std = (stDevs.get(cI) + stDevs.get(cJ)) / 2.0;
    for (VectorWritable vwI : repI) {
      if (measure.distance(uIJ, vwI.get()) <= std) {
        density++;
      }
    }
    for (VectorWritable vwJ : repJ) {
      if (measure.distance(uIJ, vwJ.get()) <= std) {
        density++;
      }
    }
    return density / (repI.size() + repJ.size());
  }

  double intraDensity(Vector clusterCenter, Vector repPoint, double avgStd) {
    return measure.distance(clusterCenter, repPoint) <= avgStd ? 1.0 : 0.0;
  }

  public double getCDbw() {
    pruneInvalidClusters();
    return intraClusterDensity() * separation();
  }

  public double intraClusterDensity() {
    pruneInvalidClusters();
    double avgStd = 0.0;
    for (Integer cId : representativePoints.keySet()) {
      avgStd += stDevs.get(cId);
    }
    avgStd /= representativePoints.size();

    double sum = 0.0;
    for (Map.Entry<Integer, List<VectorWritable>> entry : representativePoints.entrySet()) {
      Integer cId = entry.getKey();
      List<VectorWritable> repI = entry.getValue();
      double cSum = 0.0;
      for (VectorWritable aRepI : repI) {
        double inDensity = intraDensity(clusters.get(cId).getCenter(), aRepI.get(), avgStd);
        double std = stDevs.get(cId);
        if (std > 0.0) {
          cSum += inDensity / std;
        }
      }
      if (repI.size() > 0) {
        sum += cSum / repI.size();
      }
    }
    return sum / representativePoints.size();
  }

  public double interClusterDensity() {
    pruneInvalidClusters();
    double sum = 0.0;
    for (Map.Entry<Integer, List<VectorWritable>> entry1 : representativePoints.entrySet()) {
      Integer cI = entry1.getKey();
      List<VectorWritable> repI = entry1.getValue();
      for (Map.Entry<Integer, List<VectorWritable>> entry2 : representativePoints.entrySet()) {
        Integer cJ = entry2.getKey();
        if (cI.equals(cJ)) {
          continue;
        }
        List<VectorWritable> repJ = entry2.getValue();
        double minDistance = Double.MAX_VALUE;
        Vector uIJ = null;
        for (VectorWritable aRepI : repI) {
          for (VectorWritable aRepJ : repJ) {
            Vector vI = aRepI.get();
            Vector vJ = aRepJ.get();
            double distance = measure.distance(vI, vJ);
            if (distance < minDistance) {
              minDistance = distance;
              uIJ = vI.plus(vJ).divide(2);
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

        log.debug("minDistance[" + cI + "," + cJ + "]=" + minDistance);
        log.debug("stDev[" + cI + "]=" + stDevI);
        log.debug("stDev[" + cJ + "]=" + stDevJ);
        log.debug("interDensity[" + cI + "," + cJ + "]=" + interDensity);
        log.debug("density[" + cI + "," + cJ + "]=" + density);

        sum += density;
      }
    }
    //System.out.println("interClusterDensity=" + sum);
    return sum;
  }

  public double separation() {
    pruneInvalidClusters();
    double minDistance = Double.MAX_VALUE;
    for (Map.Entry<Integer, List<VectorWritable>> entry1 : representativePoints.entrySet()) {
      Integer cI = entry1.getKey();
      List<VectorWritable> repI = entry1.getValue();
      for (Map.Entry<Integer, List<VectorWritable>> entry2 : representativePoints.entrySet()) {
        if (cI.equals(entry2.getKey())) {
          continue;
        }
        List<VectorWritable> repJ = entry2.getValue();
        for (VectorWritable aRepI : repI) {
          for (VectorWritable aRepJ : repJ) {
            double distance = measure.distance(aRepI.get(), aRepJ.get());
            if (distance < minDistance) {
              minDistance = distance;
            }
          }
        }
      }
    }
    return minDistance / (1.0 + interClusterDensity());
  }
}
