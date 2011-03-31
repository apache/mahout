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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;

class DisplayKMeans extends DisplayClustering {

  //static List<List<Cluster>> result;

  DisplayKMeans() {
    initialize();
    this.setTitle("k-Means Clusters (>" + (int) (significance * 100) + "% of population)");
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
    //boolean b = true;
    int maxIter = 10;
    double distanceThreshold = 0.001;
    //if (b) {
    Path clusters = RandomSeedGenerator.buildRandom(samples, new Path(output, "clusters-0"), 3, measure);
    KMeansDriver.run(samples,
                     clusters,
                     output,
                     measure,
                     distanceThreshold,
                     maxIter,
                     true,
                     true);
    loadClusters(output);
    //} else {
    //  List<Vector> points = new ArrayList<Vector>();
    //  for (VectorWritable sample : SAMPLE_DATA) {
    //    points.add(sample.get());
    //  }
    //  List<Cluster> initialClusters = new ArrayList<Cluster>();
    //  int id = 0;
    //  int numClusters = 3;
    //  for (Vector point : points) {
    //    if (initialClusters.size() < Math.min(numClusters, points.size())) {
    //      initialClusters.add(new Cluster(point, id++));
    //    } else {
    //      break;
    //    }
    //  }
    //
    //  result = KMeansClusterer.clusterPoints(points, initialClusters, measure, maxIter, distanceThreshold);
    //  for (List<Cluster> models : result) {
    //    List<org.apache.mahout.clustering.Cluster> clusters = new ArrayList<org.apache.mahout.clustering.Cluster>();
    //    for (AbstractCluster cluster : models) {
    //      org.apache.mahout.clustering.Cluster cluster2 = (org.apache.mahout.clustering.Cluster) cluster;
    //      if (isSignificant(cluster2)) {
    //        clusters.add(cluster2);
    //      }
    //    }
    //    CLUSTERS.add(clusters);
    //  }
    //}
    new DisplayKMeans();
  }

  // Override the paint() method
  @Override
  public void paint(Graphics g) {
    plotSampleData((Graphics2D) g);
    plotClusters((Graphics2D) g);
  }
}
