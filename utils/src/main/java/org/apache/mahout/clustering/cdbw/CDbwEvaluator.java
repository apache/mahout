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
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.SquareRootFunction;

public class CDbwEvaluator {

  private final Map<Integer, List<VectorWritable>> representativePoints;

  private final Map<Integer, Double> stDevs = new HashMap<Integer, Double>();

  private final Map<Integer, Cluster> clusters;

  private final DistanceMeasure measure;

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
      setStDev(cId);
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
  public CDbwEvaluator(Configuration conf, Path clustersIn)
      throws ClassNotFoundException, InstantiationException, IllegalAccessException, IOException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    measure = ccl.loadClass(conf.get(CDbwDriver.DISTANCE_MEASURE_KEY))
        .asSubclass(DistanceMeasure.class).newInstance();
    representativePoints = CDbwMapper.getRepresentativePoints(conf);
    clusters = loadClusters(conf, clustersIn);
    for (Integer cId : representativePoints.keySet()) {
      setStDev(cId);
    }
  }

  public double getCDbw() {
    return intraClusterDensity() * separation();
  }

  public double intraClusterDensity() {
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
      sum += cSum / repI.size();
    }
    return sum / representativePoints.size();
  }

  public double interClusterDensity() {
    double sum = 0.0;
    for (Map.Entry<Integer, List<VectorWritable>> entry1 : representativePoints.entrySet()) {
      Integer cI = entry1.getKey();
      List<VectorWritable> repI = entry1.getValue();
      double stDevI = stDevs.get(cI);      
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
        double stDevJ = stDevs.get(cJ);
        double interDensity = interDensity(uIJ, cI, cJ);
        double stdSum = stDevI + stDevJ;
        double density = 0.0;
        if (stdSum > 0.0) {
          density = minDistance * interDensity / stdSum;
        }
  
        // Use a logger
        //if (false) {
        //  System.out.println("minDistance[" + cI + "," + cJ + "]=" + minDistance);
        //  System.out.println("stDev[" + cI + "]=" + stDevI);
        //  System.out.println("stDev[" + cJ + "]=" + stDevJ);
        //  System.out.println("interDensity[" + cI + "," + cJ + "]=" + interDensity);
        //  System.out.println("density[" + cI + "," + cJ + "]=" + density);
        //  System.out.println();
        //}
        sum += density;
      }
    }
    //System.out.println("interClusterDensity=" + sum);
    return sum;
  }

  public double separation() {
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

  /**
   * Load the clusters from their sequence files
   * 
   * @param clustersIn 
   *            a String pathname to the directory containing input cluster files
   * @return a List<Cluster> of the clusters
   */
  private static Map<Integer, Cluster> loadClusters(Configuration conf, Path clustersIn)
      throws InstantiationException, IllegalAccessException, IOException {
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

  private void setStDev(int cI) {
    List<VectorWritable> repPts = representativePoints.get(cI);
    //if (repPts == null) {
    //  System.out.println();
    //}
    int s0 = 0;
    Vector s1 = null;
    Vector s2 = null;
    for (VectorWritable vw : repPts) {
      s0++;
      Vector v = vw.get();
      s1 = s1 == null ? v.clone() : s1.plus(v);
      s2 = s2 == null ? v.times(v) : s2.plus(v.times(v));
    }
    Vector std = s2.times(s0).minus(s1.times(s1)).assign(new SquareRootFunction()).divide(s0);
    double d = std.zSum() / std.size();
    //System.out.println("stDev[" + cI + "]=" + d);
    stDevs.put(cI, d);
  }

  /*
  double minRpDistance(Iterable<VectorWritable> repI, Iterable<VectorWritable> repJ) {
    double minDistance = Double.MAX_VALUE;
    for (VectorWritable aRepI : repI) {
      for (VectorWritable aRepJ : repJ) {
        double distance = measure.distance(aRepI.get(), aRepJ.get());
        if (distance < minDistance) {
          minDistance = distance;
        }
      }
    }
    return minDistance;
  }
   */

  double intraDensity(Vector clusterCenter, Vector repPoint, double avgStd) {
    return measure.distance(clusterCenter, repPoint) <= avgStd ? 1.0 : 0.0;
  }
}
