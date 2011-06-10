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

package org.apache.mahout.clustering.display;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusterClassifier;
import org.apache.mahout.clustering.ClusterIterator;
import org.apache.mahout.clustering.ClusteringPolicy;
import org.apache.mahout.clustering.FuzzyKMeansClusteringPolicy;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.Vector;

public class DisplayFuzzyKMeans extends DisplayClustering {
  
  DisplayFuzzyKMeans() {
    initialize();
    this.setTitle("Fuzzy k-Means Clusters (>" + (int) (significance * 100)
        + "% of population)");
  }
  
  // Override the paint() method
  @Override
  public void paint(Graphics g) {
    plotSampleData((Graphics2D) g);
    plotClusters((Graphics2D) g);
  }
  
  public static void main(String[] args) throws Exception {
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    
    Path samples = new Path("samples");
    Path output = new Path("output");
    Configuration conf = new Configuration();
    HadoopUtil.delete(conf, samples);
    HadoopUtil.delete(conf, output);
    RandomUtils.useTestSeed();
    DisplayClustering.generateSamples();
    writeSampleData(samples);
    boolean runClusterer = false;
    int maxIterations = 10;
    if (runClusterer) {
      runSequentialFuzzyKClusterer(conf, samples, output, measure, maxIterations);
    } else {
      int numClusters = 3;
      runSequentialFuzzyKClassifier(conf, samples, output, measure, numClusters, maxIterations);
    }
    new DisplayFuzzyKMeans();
  }
  
  private static void runSequentialFuzzyKClassifier(Configuration conf,
                                                    Path samples,
                                                    Path output,
                                                    DistanceMeasure measure,
                                                    int numClusters,
                                                    int maxIterations) throws IOException {
    Collection<Vector> points = Lists.newArrayList();
    for (int i = 0; i < numClusters; i++) {
      points.add(SAMPLE_DATA.get(i).get());
    }
    List<Cluster> initialClusters = Lists.newArrayList();
    int id = 0;
    for (Vector point : points) {
      initialClusters.add(new SoftCluster(point, id++, measure));
    }
    ClusterClassifier prior = new ClusterClassifier(initialClusters);
    Path priorClassifier = new Path(output, "classifier-0");
    writeClassifier(prior, conf, priorClassifier);
    
    ClusteringPolicy policy = new FuzzyKMeansClusteringPolicy();
    new ClusterIterator(policy).iterate(samples, priorClassifier, output, maxIterations);
    for (int i = 1; i <= maxIterations; i++) {
      ClusterClassifier posterior = readClassifier(conf, new Path(output, "classifier-" + i));
      CLUSTERS.add(posterior.getModels());
    }
  }
  
  private static void runSequentialFuzzyKClusterer(Configuration conf,
                                                   Path samples,
                                                   Path output,
                                                   DistanceMeasure measure,
                                                   int maxIterations)
    throws IOException, ClassNotFoundException, InterruptedException {
    Path clusters = RandomSeedGenerator.buildRandom(conf, samples, new Path(
        output, "clusters-0"), 3, measure);
    double threshold = 0.001;
    int m = 3;
    FuzzyKMeansDriver.run(samples, clusters, output, measure, threshold,
        maxIterations, m, true, true, threshold, true);
    
    loadClusters(output);
  }
}
