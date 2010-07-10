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
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.meanshift.MeanShiftCanopy;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopyClusterer;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class DisplayMeanShift extends DisplayClustering {

  private static final Logger log = LoggerFactory.getLogger(DisplayMeanShift.class);

  private static List<MeanShiftCanopy> canopies = new ArrayList<MeanShiftCanopy>();

  private static double t1;

  private static double t2;

  private DisplayMeanShift() {
    initialize();
    this.setTitle("k-Means Clusters (>" + (int) (SIGNIFICANCE * 100) + "% of population)");
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
    for (MeanShiftCanopy canopy : canopies) {
      if (canopy.getBoundPoints().toList().size() >= SIGNIFICANCE * DisplayClustering.SAMPLE_DATA.size()) {
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

  public static void main(String[] args) {
    t1 = 1.5;
    t2 = 0.1;
    SIGNIFICANCE = 0.02;

    RandomUtils.useTestSeed();
    DisplayClustering.generateSamples();
    List<Vector> points = new ArrayList<Vector>();
    for (VectorWritable sample : SAMPLE_DATA) {
      points.add(sample.get());
    }
    canopies = MeanShiftCanopyClusterer.clusterPoints(points, new EuclideanDistanceMeasure(), 0.005, t1, t2, 20);
    for (MeanShiftCanopy canopy : canopies) {
      log.info(canopy.toString());
    }
    new DisplayMeanShift();
  }
}
