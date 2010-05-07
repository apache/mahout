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

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.SquareRootFunction;

public class CDbwEvaluator {

  Map<Integer, List<VectorWritable>> representativePoints;

  Map<Integer, Double> stDevs = new HashMap<Integer, Double>();

  Map<Integer, Cluster> clusters;

  DistanceMeasure measure;

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
  public CDbwEvaluator(Map<Integer, List<VectorWritable>> representativePoints, Map<Integer, Cluster> clusters,
      DistanceMeasure measure) {
    super();
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
   * @param job
   *            a JobConf with appropriate parameters
   * @param clustersIn
   *            a String path to the input clusters directory
   *            
   * @throws SecurityException
   * @throws IllegalArgumentException
   * @throws NoSuchMethodException
   * @throws InvocationTargetException
   * @throws ClassNotFoundException
   * @throws InstantiationException
   * @throws IllegalAccessException
   * @throws IOException
   */
  public CDbwEvaluator(JobConf job, Path clustersIn)
      throws SecurityException, IllegalArgumentException, NoSuchMethodException,
      InvocationTargetException, ClassNotFoundException, InstantiationException, IllegalAccessException, IOException {
    super();
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<?> cl = ccl.loadClass(job.get(CDbwDriver.DISTANCE_MEASURE_KEY));
    measure = (DistanceMeasure) cl.newInstance();
    representativePoints = CDbwMapper.getRepresentativePoints(job);
    clusters = loadClusters(job, clustersIn);
    for (Integer cId : representativePoints.keySet()) {
      setStDev(cId);
    }
  }

  public double CDbw() {
    double cdbw = intraClusterDensity() * separation();
    System.out.println("CDbw=" + cdbw);
    return cdbw;
  }

  /**
   * Load the clusters from their sequence files
   * 
   * @param clustersIn 
   *            a String pathname to the directory containing input cluster files
   * @return a List<Cluster> of the clusters
   * 
   * @throws ClassNotFoundException
   * @throws InstantiationException
   * @throws IllegalAccessException
   * @throws IOException
   * @throws SecurityException
   * @throws NoSuchMethodException
   * @throws InvocationTargetException
   */
  private HashMap<Integer, Cluster> loadClusters(JobConf job, Path clustersIn)
      throws InstantiationException, IllegalAccessException, IOException, SecurityException {
    HashMap<Integer, Cluster> clusters = new HashMap<Integer, Cluster>();
    FileSystem fs = clustersIn.getFileSystem(job);
    for (FileStatus part : fs.listStatus(clustersIn)) {
      if (!part.getPath().getName().startsWith(".")) {
        Path inPart = part.getPath();
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, inPart, job);
        Writable key = (Writable) reader.getKeyClass().newInstance();
        Writable value = (Writable) reader.getValueClass().newInstance();
        while (reader.next(key, value)) {
          Cluster cluster = (Cluster) value;
          clusters.put(cluster.getId(), cluster);
          value = (Writable) reader.getValueClass().newInstance();
        }
        reader.close();
      }
    }
    return clusters;
  }

  double interClusterDensity() {
    double sum = 0;
    for (int cI : representativePoints.keySet()) {
      for (int cJ : representativePoints.keySet()) {
        if (cI == cJ) {
          continue;
        }
        List<VectorWritable> repI = representativePoints.get(cI);
        List<VectorWritable> repJ = representativePoints.get(cJ);
        double minDistance = Double.MAX_VALUE;
        Vector uIJ = null;
        for (int ptI = 0; ptI < repI.size(); ptI++) {
          for (int ptJ = 0; ptJ < repJ.size(); ptJ++) {
            Vector vI = repI.get(ptI).get();
            Vector vJ = repJ.get(ptJ).get();
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
        double density = 0;
        if (stdSum > 0) {
          density = minDistance * interDensity / stdSum;
        }

        if (false) {
          System.out.println("minDistance[" + cI + "," + cJ + "]=" + minDistance);
          System.out.println("stDev[" + cI + "]=" + stDevI);
          System.out.println("stDev[" + cJ + "]=" + stDevJ);
          System.out.println("interDensity[" + cI + "," + cJ + "]=" + interDensity);
          System.out.println("density[" + cI + "," + cJ + "]=" + density);
          System.out.println();
        }
        sum += density;
      }
    }
    System.out.println("interClusterDensity=" + sum);
    return sum;
  }

  double interDensity(Vector uIJ, int cI, int cJ) {
    List<VectorWritable> repI = representativePoints.get(cI);
    List<VectorWritable> repJ = representativePoints.get(cJ);
    double density = 0;
    double std = (stDevs.get(cI) + stDevs.get(cJ)) / 2;
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
    double d = 0;
    if (repPts == null) {
      System.out.println();
    }
    int s0 = 0;
    Vector s1 = null;
    Vector s2 = null;
    for (VectorWritable vw : repPts) {
      s0++;
      Vector v = vw.get();
      if (s1 == null) {
        s1 = v.clone();
      } else {
        s1 = s1.plus(v);
      }
      if (s2 == null) {
        s2 = v.times(v);
      } else {
        s2 = s2.plus(v.times(v));
      }
    }
    Vector std = s2.times(s0).minus(s1.times(s1)).assign(new SquareRootFunction()).divide(s0);
    d = std.zSum() / std.size();
    System.out.println("stDev[" + cI + "]=" + d);
    stDevs.put(cI, d);
  }

  double minRpDistance(List<VectorWritable> repI, List<VectorWritable> repJ) {
    double minDistance = Double.MAX_VALUE;
    for (int ptI = 0; ptI < repI.size(); ptI++) {
      for (int ptJ = 0; ptJ < repJ.size(); ptJ++) {
        double distance = measure.distance(repI.get(ptI).get(), repJ.get(ptJ).get());
        if (distance < minDistance) {
          minDistance = distance;
        }
      }
    }
    return minDistance;
  }

  double separation() {
    double minDistance = Double.MAX_VALUE;
    for (Integer cI : representativePoints.keySet()) {
      for (int cJ : representativePoints.keySet()) {
        if (cI == cJ) {
          continue;
        }
        List<VectorWritable> repI = representativePoints.get(cI);
        List<VectorWritable> repJ = representativePoints.get(cJ);
        for (int ptI = 0; ptI < repI.size(); ptI++) {
          for (int ptJ = 0; ptJ < repJ.size(); ptJ++) {
            double distance = measure.distance(repI.get(ptI).get(), repJ.get(ptJ).get());
            if (distance < minDistance) {
              minDistance = distance;
            }
          }
        }
      }
    }
    double separation = minDistance / (1 + interClusterDensity());
    System.out.println("separation=" + separation);
    return separation;
  }

  double intraClusterDensity() {
    double avgStd = 0;
    for (Integer cId : representativePoints.keySet()) {
      avgStd += stDevs.get(cId);
    }
    avgStd = avgStd / representativePoints.size();

    double sum = 0;
    for (Integer cId : representativePoints.keySet()) {
      List<VectorWritable> repI = representativePoints.get(cId);
      double cSum = 0;
      for (int ptI = 0; ptI < repI.size(); ptI++) {
        double inDensity = intraDensity(clusters.get(cId).getCenter(), repI.get(ptI).get(), avgStd);
        Double std = stDevs.get(cId);
        if (std > 0) {
          cSum += inDensity / std;
        }
      }
      sum += cSum / repI.size();
    }
    double intraClusterDensity = sum / representativePoints.size();
    System.out.println("intraClusterDensity=" + intraClusterDensity);
    return intraClusterDensity;
  }

  double intraDensity(Vector clusterCenter, Vector repPoint, double avgStd) {
    if (measure.distance(clusterCenter, repPoint) <= avgStd) {
      return 1;
    }
    return 0;
  }
}
