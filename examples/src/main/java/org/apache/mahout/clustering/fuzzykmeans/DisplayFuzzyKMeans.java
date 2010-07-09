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

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.dirichlet.DisplayClustering;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

class DisplayFuzzyKMeans extends DisplayClustering {

  private static List<List<SoftCluster>> clusters;
  
  DisplayFuzzyKMeans() {
    initialize();
    this.setTitle("Fuzzy K-Means Clusters (> 5% of population)");
  }
  
  @Override
  public void paint(Graphics g) {
    plotSampleData(g);
    Graphics2D g2 = (Graphics2D) g;
    Vector dv = new DenseVector(2);
    int i = DisplayFuzzyKMeans.clusters.size() - 1;
    for (List<SoftCluster> cls : clusters) {
      g2.setStroke(new BasicStroke(i == 0 ? 3 : 1));
      g2.setColor(COLORS[Math.min(DisplayClustering.COLORS.length - 1, i--)]);
      for (SoftCluster cluster : cls) {
        // if (true || cluster.getWeightedPointTotal().zSum() > sampleData.size() * 0.05) {
        dv.assign(Math.max(cluster.std(), 0.3) * 3);
        DisplayClustering.plotEllipse(g2, cluster.getCenter(), dv);
        // }
      }
    }
  }
  
  public static void main(String[] args) {
    RandomUtils.useTestSeed();
    DisplayClustering.generateSamples();
    List<Vector> points = new ArrayList<Vector>();
    for (VectorWritable sample : SAMPLE_DATA) {
      points.add((Vector) sample.get());
    }
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    List<SoftCluster> initialClusters = new ArrayList<SoftCluster>();
    
    k = 3;
    int i = 0;
    for (Vector point : points) {
      if (initialClusters.size() < Math.min(k, points.size())) {
        initialClusters.add(new SoftCluster(point, i++));
      } else {
        break;
      }
    }
    clusters = FuzzyKMeansClusterer.clusterPoints(points, initialClusters, measure, 0.001, 3, 10);
    new DisplayFuzzyKMeans();
  }
}
