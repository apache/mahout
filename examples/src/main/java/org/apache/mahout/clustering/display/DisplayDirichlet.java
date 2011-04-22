/*
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

package org.apache.mahout.clustering.display;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusterClassifier;
import org.apache.mahout.clustering.ClusterIterator;
import org.apache.mahout.clustering.ClusteringPolicy;
import org.apache.mahout.clustering.DirichletClusteringPolicy;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.clustering.ModelDistribution;
import org.apache.mahout.clustering.dirichlet.DirichletClusterer;
import org.apache.mahout.clustering.dirichlet.models.GaussianClusterDistribution;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DisplayDirichlet extends DisplayClustering {
  
  private static final Logger log = LoggerFactory
      .getLogger(DisplayDirichlet.class);
  
  public DisplayDirichlet() {
    initialize();
    this.setTitle("Dirichlet Process Clusters - Normal Distribution (>"
        + (int) (significance * 100) + "% of population)");
  }
  
  // Override the paint() method
  @Override
  public void paint(Graphics g) {
    plotSampleData((Graphics2D) g);
    plotClusters((Graphics2D) g);
  }
  
  protected static void printModels(Iterable<Cluster[]> result, int significant) {
    int row = 0;
    StringBuilder models = new StringBuilder(100);
    for (Cluster[] r : result) {
      models.append("sample[").append(row++).append("]= ");
      for (int k = 0; k < r.length; k++) {
        Cluster model = r[k];
        if (model.count() > significant) {
          models.append('m').append(k).append(model.asFormatString(null))
              .append(", ");
        }
      }
      models.append('\n');
    }
    models.append('\n');
    log.info(models.toString());
  }
  
  protected static void generateResults(
      ModelDistribution<VectorWritable> modelDist, int numClusters,
      int numIterations, double alpha0, int thin, int burnin) {
    boolean b = false;
    if (b) {
      DirichletClusterer dc = new DirichletClusterer(SAMPLE_DATA, modelDist,
          alpha0, numClusters, thin, burnin);
      List<Cluster[]> result = dc.cluster(numIterations);
      printModels(result, burnin);
      for (Cluster[] models : result) {
        List<Cluster> clusters = new ArrayList<Cluster>();
        for (Cluster cluster : models) {
          if (isSignificant(cluster)) {
            clusters.add(cluster);
          }
        }
        CLUSTERS.add(clusters);
      }
    } else {
      List<Vector> points = new ArrayList<Vector>();
      for (VectorWritable sample : SAMPLE_DATA) {
        points.add(sample.get());
      }
      ClusteringPolicy policy = new DirichletClusteringPolicy(numClusters,
          numIterations);
      List<Cluster> models = new ArrayList<Cluster>();
      for (Model<VectorWritable> cluster : modelDist
          .sampleFromPrior(numClusters)) {
        models.add((Cluster) cluster);
      }
      ClusterClassifier prior = new ClusterClassifier(models);
      ClusterIterator iterator = new ClusterIterator(policy);
      ClusterClassifier posterior = iterator.iterate(points, prior, 5);
      List<Cluster> models2 = posterior.getModels();
      for (Iterator<Cluster> it = models2.iterator(); it.hasNext();) {
        if (!isSignificant(it.next())) it.remove();
      }
      CLUSTERS.add(models2);
    }
  }
  
  public static void main(String[] args) throws Exception {
    VectorWritable modelPrototype = new VectorWritable(new DenseVector(2));
    ModelDistribution<VectorWritable> modelDist = new GaussianClusterDistribution(
        modelPrototype);
    
    RandomUtils.useTestSeed();
    generateSamples();
    int numIterations = 20;
    int numClusters = 10;
    int alpha0 = 1;
    int thin = 3;
    int burnin = 5;
    generateResults(modelDist, numClusters, numIterations, alpha0, thin, burnin);
    new DisplayDirichlet();
  }
  
}
