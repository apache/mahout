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

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopy;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopyDriver;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.kernel.IKernelProfile;
import org.apache.mahout.common.kernel.TriangularKernelProfile;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class DisplayMeanShift extends DisplayClustering {
  
  private static double t1;
  
  private static double t2;
  
  private DisplayMeanShift() {
    initialize();
    this.setTitle("Mean Shift Canopy Clusters (>" + (int) (significance * 100)
        + "% of population)");
  }
  
  @Override
  public void paint(Graphics g) {
    Graphics2D g2 = (Graphics2D) g;
    double sx = (double) res / DS;
    g2.setTransform(AffineTransform.getScaleInstance(sx, sx));
    
    // plot the axes
    g2.setColor(Color.BLACK);
    Vector dv = new DenseVector(2).assign(SIZE / 2.0);
    Vector dv1 = new DenseVector(2).assign(t1);
    Vector dv2 = new DenseVector(2).assign(t2);
    DisplayClustering.plotRectangle(g2, new DenseVector(2).assign(2), dv);
    DisplayClustering.plotRectangle(g2, new DenseVector(2).assign(-2), dv);
    
    // plot the sample data
    g2.setColor(Color.DARK_GRAY);
    dv.assign(0.03);
    for (VectorWritable v : SAMPLE_DATA) {
      DisplayClustering.plotRectangle(g2, v.get(), dv);
    }
    int i = 0;
    for (Cluster cluster : CLUSTERS.get(CLUSTERS.size() - 1)) {
      MeanShiftCanopy canopy = (MeanShiftCanopy) cluster;
      if (canopy.getMass() >= significance
          * DisplayClustering.SAMPLE_DATA.size()) {
        g2.setColor(COLORS[Math.min(i++, DisplayClustering.COLORS.length - 1)]);
        int count = 0;
        Vector center = new DenseVector(2);
        for (int vix : canopy.getBoundPoints().toList()) {
          Vector v = SAMPLE_DATA.get(vix).get();
          count++;
          v.addTo(center);
          DisplayClustering.plotRectangle(g2, v, dv);
        }
        center = center.divide(count);
        DisplayClustering.plotEllipse(g2, center, dv1);
        DisplayClustering.plotEllipse(g2, center, dv2);
      }
    }
  }
  
  public static void main(String[] args) throws Exception {
    t1 = 1.5;
    t2 = 0.5;
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    IKernelProfile kernelProfile = new TriangularKernelProfile();
    significance = 0.02;
    
    Path samples = new Path("samples");
    Path output = new Path("output");
    Configuration conf = new Configuration();
    HadoopUtil.delete(conf, samples);
    HadoopUtil.delete(conf, output);
    
    RandomUtils.useTestSeed();
    DisplayClustering.generateSamples();
    writeSampleData(samples);
    // boolean b = true;
    // if (b) {
    MeanShiftCanopyDriver.run(conf, samples, output, measure, kernelProfile,
        t1, t2, 0.005, 20, false, true, true);
    loadClusters(output);
    // } else {
    // Collection<Vector> points = new ArrayList<Vector>();
    // for (VectorWritable sample : SAMPLE_DATA) {
    // points.add(sample.get());
    // }
    // List<MeanShiftCanopy> canopies =
    // MeanShiftCanopyClusterer.clusterPoints(points, measure, 0.005, t1, t2,
    // 20);
    // for (MeanShiftCanopy canopy : canopies) {
    // log.info(canopy.toString());
    // }
    // }
    new DisplayMeanShift();
  }
}
