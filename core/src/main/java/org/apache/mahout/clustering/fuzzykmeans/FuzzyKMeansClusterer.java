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
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.ClusterObservations;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class FuzzyKMeansClusterer {

  private static final double MINIMAL_VALUE = 0.0000000001;

  private DistanceMeasure measure;
  private double convergenceDelta;
  private double m = 2.0; // default value
  private boolean emitMostLikely = true;
  private double threshold;

  /**
    * Init the fuzzy k-means clusterer with the distance measure to use for comparison.
   */
  public FuzzyKMeansClusterer(DistanceMeasure measure, double convergenceDelta, double m) {
    this.measure = measure;
    this.convergenceDelta = convergenceDelta;
    this.m = m;
  }

  public FuzzyKMeansClusterer(Configuration conf) {
    this.configure(conf);
  }

  public FuzzyKMeansClusterer() {
  }

  /**
   * This is the reference k-means implementation. Given its inputs it iterates over the points and clusters
   * until their centers converge or until the maximum number of iterations is exceeded.
   * 
   * @param points
   *          the input List<Vector> of points
   * @param clusters
   *          the initial List<SoftCluster> of clusters
   * @param measure
   *          the DistanceMeasure to use
   * @param threshold
   *          the double convergence threshold
   * @param m
   *          the double "fuzzyness" argument (>1)
   * @param numIter
   *          the maximum number of iterations
   * @return
   *          a List<List<SoftCluster>> of clusters produced per iteration
   */
  public static List<List<SoftCluster>> clusterPoints(Iterable<Vector> points,
                                                      List<SoftCluster> clusters,
                                                      DistanceMeasure measure,
                                                      double threshold,
                                                      double m,
                                                      int numIter) {
    List<List<SoftCluster>> clustersList = Lists.newArrayList();
    clustersList.add(clusters);
    FuzzyKMeansClusterer clusterer = new FuzzyKMeansClusterer(measure, threshold, m);
    boolean converged = false;
    int iteration = 0;
    for (int iter = 0; !converged && iter < numIter; iter++) {
      List<SoftCluster> next = Lists.newArrayList();
      List<SoftCluster> cs = clustersList.get(iteration++);
      for (SoftCluster c : cs) {
        next.add(new SoftCluster(c.getCenter(), c.getId(), measure));
      }
      clustersList.add(next);
      converged = runFuzzyKMeansIteration(points, clustersList.get(iteration), clusterer);
    }
    return clustersList;
  }

  /**
   * Perform a single iteration over the points and clusters, assigning points to clusters and returning if
   * the iterations are completed.
   * 
   * @param points
   *          the List<Vector> having the input points
   * @param clusterList
   *          the List<Cluster> clusters
   */
  protected static boolean runFuzzyKMeansIteration(Iterable<Vector> points,
                                                   List<SoftCluster> clusterList,
                                                   FuzzyKMeansClusterer clusterer) {
    for (Vector point : points) {
      clusterer.addPointToClusters(clusterList, point);
    }
    return clusterer.testConvergence(clusterList);
  }

  /**
   * Configure the distance measure from the job
   */
  private void configure(Configuration job) {
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      measure = ccl.loadClass(job.get(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY))
          .asSubclass(DistanceMeasure.class).newInstance();
      measure.configure(job);
      convergenceDelta = Double.parseDouble(job.get(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY));
      // nextClusterId = 0;
      m = Double.parseDouble(job.get(FuzzyKMeansConfigKeys.M_KEY));
      emitMostLikely = Boolean.parseBoolean(job.get(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY));
      threshold = Double.parseDouble(job.get(FuzzyKMeansConfigKeys.THRESHOLD_KEY));

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
   * @param point
   *          a point
   * @param clusters
   *          a List<SoftCluster>
   * @param context
   *          the Context to emit into
   */
  public void emitPointProbToCluster(Vector point,
                                     List<SoftCluster> clusters,
                                     Mapper<?,?,Text,ClusterObservations>.Context context)
    throws IOException, InterruptedException {

    List<Double> clusterDistanceList = Lists.newArrayList();
    for (SoftCluster cluster : clusters) {
      clusterDistanceList.add(measure.distance(cluster.getCenter(), point));
    }

    for (int i = 0; i < clusters.size(); i++) {
      SoftCluster cluster = clusters.get(i);
      Text key = new Text(cluster.getIdentifier());
      ClusterObservations value =
          new ClusterObservations(computeProbWeight(clusterDistanceList.get(i), clusterDistanceList),
                                  point,
                                  point.times(point));
      context.write(key, value);
    }
  }

  /** Computes the probability of a point belonging to a cluster */
  public double computeProbWeight(double clusterDistance, Iterable<Double> clusterDistanceList) {
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
  public boolean computeConvergence(Cluster cluster) {
    return cluster.computeConvergence(measure, convergenceDelta);
  }

  public double getM() {
    return m;
  }

  public DistanceMeasure getMeasure() {
    return this.measure;
  }

  public void emitPointToClusters(VectorWritable point,
                                  List<SoftCluster> clusters,
                                  Mapper<?,?,IntWritable,WeightedVectorWritable>.Context context)
    throws IOException, InterruptedException {
    // calculate point distances for all clusters    
    List<Double> clusterDistanceList = Lists.newArrayList();
    for (SoftCluster cluster : clusters) {
      clusterDistanceList.add(getMeasure().distance(cluster.getCenter(), point.get()));
    }
    // calculate point pdf for all clusters
    Vector pi = computePi(clusters, clusterDistanceList);
    if (emitMostLikely) {
      emitMostLikelyCluster(point.get(), clusters, pi, context);
    } else {
      emitAllClusters(point.get(), clusters, pi, context);
    }
  }

  public Vector computePi(Collection<SoftCluster> clusters, List<Double> clusterDistanceList) {
    Vector pi = new DenseVector(clusters.size());
    for (int i = 0; i < clusters.size(); i++) {
      double probWeight = computeProbWeight(clusterDistanceList.get(i), clusterDistanceList);
      pi.set(i, probWeight);
    }
    return pi;
  }

  /**
   * Emit the point to the cluster with the highest pdf
   */
  private void emitMostLikelyCluster(Vector point,
                                     List<SoftCluster> clusters,
                                     Vector pi,
                                     Mapper<?,?,IntWritable,WeightedVectorWritable>.Context context)
    throws IOException, InterruptedException {
    int clusterId = -1;
    double clusterPdf = 0.0;
    for (int i = 0; i < clusters.size(); i++) {
      // System.out.println("cluster-" + clusters.get(i).getId() + "@ " + ClusterBase.formatVector(center, null));
      double pdf = pi.get(i);
      if (pdf > clusterPdf) {
        clusterId = clusters.get(i).getId();
        clusterPdf = pdf;
      }
    }
    // System.out.println("cluster-" + clusterId + ": " + ClusterBase.formatVector(point, null));
    context.write(new IntWritable(clusterId), new WeightedVectorWritable(clusterPdf, point));
  }

  /**
   * Emit the point to all clusters
   */
  private void emitAllClusters(Vector point,
                               Collection<SoftCluster> clusters,
                               Vector pi,
                               Mapper<?,?,IntWritable,WeightedVectorWritable>.Context context)
    throws IOException, InterruptedException {
    for (int i = 0; i < clusters.size(); i++) {
      double pdf = pi.get(i);
      if (pdf > threshold) {
        // System.out.println("cluster-" + clusterId + ": " + ClusterBase.formatVector(point, null));
        context.write(new IntWritable(i), new WeightedVectorWritable(pdf, point));
      }
    }
  }

  protected void addPointToClusters(List<SoftCluster> clusterList, Vector point) {
    List<Double> clusterDistanceList = Lists.newArrayList();
    for (SoftCluster cluster : clusterList) {
      clusterDistanceList.add(getMeasure().distance(point, cluster.getCenter()));
    }

    for (int i = 0; i < clusterList.size(); i++) {
      double probWeight = computeProbWeight(clusterDistanceList.get(i), clusterDistanceList);
      clusterList.get(i).observe(point, Math.pow(probWeight, getM()));
    }
  }

  protected boolean testConvergence(Iterable<SoftCluster> clusters) {
    boolean converged = true;
    for (SoftCluster cluster : clusters) {
      if (!cluster.computeConvergence(measure, convergenceDelta)) {
        converged = false;
      }
      cluster.computeParameters();
    }
    return converged;
  }

  public void emitPointToClusters(VectorWritable point, List<SoftCluster> clusters, Writer writer) throws IOException {
    // calculate point distances for all clusters    
    List<Double> clusterDistanceList = Lists.newArrayList();
    for (SoftCluster cluster : clusters) {
      clusterDistanceList.add(getMeasure().distance(cluster.getCenter(), point.get()));
    }
    Vector pi = computePi(clusters, clusterDistanceList);
    if (emitMostLikely) {
      emitMostLikelyCluster(point.get(), clusters, pi, writer);
    } else {
      emitAllClusters(point.get(), clusters, pi, writer);
    }
  }

  private void emitAllClusters(Vector point, Collection<SoftCluster> clusters, Vector pi, Writer writer)
    throws IOException {
    for (int i = 0; i < clusters.size(); i++) {
      double pdf = pi.get(i);
      if (pdf > threshold) {
        // System.out.println("cluster-" + clusterId + ": " + ClusterBase.formatVector(point, null));
        writer.append(new IntWritable(i), new WeightedVectorWritable(pdf, point));
      }
    }
  }

  private static void emitMostLikelyCluster(Vector point, List<SoftCluster> clusters, Vector pi, Writer writer)
    throws IOException {
    int clusterId = -1;
    double clusterPdf = 0.0;
    for (int i = 0; i < clusters.size(); i++) {
      // System.out.println("cluster-" + clusters.get(i).getId() + "@ " + ClusterBase.formatVector(center, null));
      double pdf = pi.get(i);
      if (pdf > clusterPdf) {
        clusterId = clusters.get(i).getId();
        clusterPdf = pdf;
      }
    }
    // System.out.println("cluster-" + clusterId + ": " + ClusterBase.formatVector(point, null));
    writer.append(new IntWritable(clusterId), new WeightedVectorWritable(clusterPdf, point));
  }
}
