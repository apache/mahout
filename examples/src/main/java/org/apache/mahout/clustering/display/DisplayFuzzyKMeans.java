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
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansClusterer;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

class DisplayFuzzyKMeans extends DisplayClustering {

  DisplayFuzzyKMeans() {
    initialize();
    this.setTitle("Fuzzy k-Means Clusters (>" + (int) (SIGNIFICANCE * 100) + "% of population)");
  }

  // Override the paint() method
  @Override
  public void paint(Graphics g) {
    plotSampleData((Graphics2D) g);
    plotClusters((Graphics2D) g);
  }

  public static void main(String[] args) {
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    double threshold = 0.001;
    int numClusters = 3;
    int numIterations = 10;
    int m = 3;
    
    RandomUtils.useTestSeed();
    DisplayClustering.generateSamples();
    List<Vector> points = new ArrayList<Vector>();
    for (VectorWritable sample : SAMPLE_DATA) {
      points.add((Vector) sample.get());
    }
    int id = 0;
    List<SoftCluster> initialClusters = new ArrayList<SoftCluster>();
    for (Vector point : points) {
      if (initialClusters.size() < Math.min(numClusters, points.size())) {
        initialClusters.add(new SoftCluster(point, id++));
      } else {
        break;
      }
    }
    List<List<SoftCluster>> results = FuzzyKMeansClusterer.clusterPoints(points,
                                                                         initialClusters,
                                                                         measure,
                                                                         threshold,
                                                                         m,
                                                                         numIterations);
    for (List<SoftCluster> models : results) {
      List<org.apache.mahout.clustering.Cluster> clusters = new ArrayList<org.apache.mahout.clustering.Cluster>();
      for (SoftCluster cluster : models) {
        org.apache.mahout.clustering.Cluster cluster2 = (org.apache.mahout.clustering.Cluster) cluster;
        if (isSignificant(cluster2)) {
          clusters.add(cluster2);
        }
      }
      CLUSTERS.add(clusters);
    }
    new DisplayFuzzyKMeans();
  }
}
