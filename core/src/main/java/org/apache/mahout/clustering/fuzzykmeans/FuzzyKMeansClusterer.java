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

package org.apache.mahout.clustering.fuzzykmeans;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;

public class FuzzyKMeansClusterer {

  private static final double MINIMAL_VALUE = 0.0000000001; // using it for
  // adding
  // exception
  // this value to any
  // zero valued
  // variable to avoid
  // divide by Zero

  //private int nextClusterId = 0;
  
  private DistanceMeasure measure;

  private double convergenceDelta = 0;
  
  private double m = 2.0; // default value
  
  /**
   * Init the fuzzy k-means clusterer with the distance measure to use for comparison.
   * 
   * @param measure
   *          The distance measure to use for comparing clusters against points.
   * @param convergenceDelta
   *          When do we define a cluster to have converged?
   * 
   * */
  public FuzzyKMeansClusterer(DistanceMeasure measure, double convergenceDelta, double m) {
    this.measure = measure;
    this.convergenceDelta = convergenceDelta;
    this.m = m;
  }
  
  public FuzzyKMeansClusterer(JobConf job) {
    this.configure(job);
  }
  
  /**
   * Configure the distance measure directly. Used by unit tests.
   *
   * @param aMeasure          the DistanceMeasure
   * @param aConvergenceDelta the delta value used to define convergence
   */
  private void config(DistanceMeasure aMeasure, double aConvergenceDelta) {
    measure = aMeasure;
    convergenceDelta = aConvergenceDelta;
    //nextClusterId = 0;
  }
  
  /**
   * Configure the distance measure from the job
   *
   * @param job the JobConf for the job
   */
  private void configure(JobConf job) {
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      Class<?> cl = ccl.loadClass(job.get(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY));
      measure = (DistanceMeasure) cl.newInstance();
      measure.configure(job);
      convergenceDelta = Double.parseDouble(job.get(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY));
      //nextClusterId = 0;
      m = Double.parseDouble(job.get(FuzzyKMeansConfigKeys.M_KEY));
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
  }
  
  /**
   * Emit the point and its probability of belongingness to each cluster
   *
   * @param point    a point
   * @param clusters a List<SoftCluster>
   * @param output   the OutputCollector to emit into
   */
  public void emitPointProbToCluster(Vector point,
    List<SoftCluster> clusters,
    OutputCollector<Text, FuzzyKMeansInfo> output) throws IOException {
    
    List<Double> clusterDistanceList = new ArrayList<Double>();
    for (SoftCluster cluster : clusters) {
      clusterDistanceList.add(measure.distance(cluster.getCenter(), point));
    }

    for (int i = 0; i < clusters.size(); i++) {
      double probWeight = computeProbWeight(clusterDistanceList.get(i), clusterDistanceList);
      Text key = new Text(clusters.get(i).getIdentifier()); // just output the
      // identifier,avoids
      // too much data
      // traffic
      /*Text value = new Text(Double.toString(probWeight)
          + FuzzyKMeansDriver.MAPPER_VALUE_SEPARATOR + values.toString());*/
      FuzzyKMeansInfo value = new FuzzyKMeansInfo(probWeight, point);
      output.collect(key, value);
    }
  }

  /**
   * Output point with cluster info (Cluster and probability)
   *
   * @param point    a point
   * @param clusters a List<SoftCluster> to test
   * @param output   the OutputCollector to emit into
   */
  public void outputPointWithClusterProbabilities(String key,
    Vector point, List<SoftCluster> clusters,
    OutputCollector<Text, FuzzyKMeansOutput> output) throws IOException {
    
    List<Double> clusterDistanceList = new ArrayList<Double>();

    for (SoftCluster cluster : clusters) {
      clusterDistanceList.add(measure.distance(point, cluster.getCenter()));
    }
    FuzzyKMeansOutput fOutput = new FuzzyKMeansOutput(clusters.size());
    for (int i = 0; i < clusters.size(); i++) {
      double probWeight = computeProbWeight(clusterDistanceList.get(i),
          clusterDistanceList);
      fOutput.add(i, clusters.get(i), probWeight);
    }
    String name = point.getName();
    output.collect(new Text(name != null && name.length() != 0 ? name
        : point.asFormatString()),
        fOutput);
  }

  /** Computes the probability of a point belonging to a cluster */
  public double computeProbWeight(double clusterDistance, List<Double> clusterDistanceList) {
    if (clusterDistance == 0) {
      clusterDistance = MINIMAL_VALUE;
    }
    double denom = 0.0;
    for (double eachCDist : clusterDistanceList) {
      if (eachCDist == 0.0) {
        eachCDist = MINIMAL_VALUE;
      }

      denom += Math.pow(clusterDistance / eachCDist, 2.0 / (m - 1));

    }
    return 1.0 / denom;
  }
  
  /**
   * Return if the cluster is converged by comparing its center and centroid.
   *
   * @return if the cluster is converged
   */
  public boolean computeConvergence(SoftCluster cluster) {
    Vector centroid = cluster.computeCentroid();
    cluster.setConverged(measure.distance(cluster.getCenter(), centroid) <= convergenceDelta);
    return cluster.isConverged();
  }
  
  public double getM() {
    return m;
  }
  
  public DistanceMeasure getMeasure() {
    return this.measure;
  }
}
