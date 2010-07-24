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
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.canopy.CanopyClusterer;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

class DisplayCanopy extends DisplayClustering {

  private static final long serialVersionUID = 1L;

  DisplayCanopy() {
    initialize();
    this.setTitle("Canopy Clusters (>" + (int) (SIGNIFICANCE * 100) + "% of population)");
  }

  @Override
  public void paint(Graphics g) {
    plotSampleData((Graphics2D) g);
    plotClusters((Graphics2D) g);
  }

  public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException, InstantiationException,
      IllegalAccessException {
    SIGNIFICANCE = 0.05;
    Path samples = new Path("samples");
    Path output = new Path("output");
    HadoopUtil.overwriteOutput(samples);
    HadoopUtil.overwriteOutput(output);
    RandomUtils.useTestSeed();
    generateSamples();
    writeSampleData(samples);
    boolean b = true;
    if (b) {
      new CanopyDriver().buildClusters(samples, output, ManhattanDistanceMeasure.class.getName(), T1, T2, true);
      loadClusters(output);
    } else {
      List<Vector> points = new ArrayList<Vector>();
      for (VectorWritable sample : SAMPLE_DATA) {
        points.add(sample.get());
      }
      List<Canopy> canopies = CanopyClusterer.createCanopies(points, new ManhattanDistanceMeasure(), T1, T2);
      CanopyClusterer.updateCentroids(canopies);
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (Canopy canopy : canopies)
        clusters.add(canopy);
      CLUSTERS.add(clusters);
    }

    new DisplayCanopy();
  }

}
