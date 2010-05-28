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

package org.apache.mahout.clustering.kmeans;

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.dirichlet.DisplayDirichlet;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

class DisplayKMeans extends DisplayDirichlet {

  private static List<List<Cluster>> clusters;

  DisplayKMeans() {
    initialize();
    this.setTitle("K-Means Clusters (> 5% of population)");
  }

  @Override
  public void paint(Graphics g) {
    super.plotSampleData(g);
    Graphics2D g2 = (Graphics2D) g;
    Vector dv = new DenseVector(2);
    int i = DisplayKMeans.clusters.size() - 1;
    for (List<Cluster> cls : clusters) {
      g2.setStroke(new BasicStroke(i == 0 ? 3 : 1));
      g2.setColor(COLORS[Math.min(DisplayDirichlet.COLORS.length - 1, i--)]);
      for (Cluster cluster : cls) {
        // if (true || cluster.getNumPoints() > sampleData.size() * 0.05) {
        dv.assign(cluster.getStd() * 3);
        System.out.println(cluster.getCenter().asFormatString() + ' ' + dv.asFormatString());
        DisplayDirichlet.plotEllipse(g2, cluster.getCenter(), dv);
        // }
      }
    }
  }

  public static void main(String[] args) {
    RandomUtils.useTestSeed();
    DisplayDirichlet.generateSamples();
    List<Vector> points = new ArrayList<Vector>();
    for (VectorWritable sample : SAMPLE_DATA) {
      points.add(sample.get());
    }
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    List<Cluster> initialClusters = new ArrayList<Cluster>();
    k = 3;
    int i = 0;
    for (Vector point : points) {
      if (initialClusters.size() < Math.min(k, points.size())) {
        initialClusters.add(new Cluster(point, i++));
      } else {
        break;
      }
    }
    clusters = KMeansClusterer.clusterPoints(points, initialClusters, measure, 10, 0.001);
    System.out.println(clusters.size());
    new DisplayKMeans();
  }
}
