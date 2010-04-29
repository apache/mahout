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

package org.apache.mahout.clustering.meanshift;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.dirichlet.DisplayDirichlet;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class DisplayMeanShift extends DisplayDirichlet {

  private static final Logger log = LoggerFactory.getLogger(DisplayMeanShift.class);

  private static List<MeanShiftCanopy> canopies = new ArrayList<MeanShiftCanopy>();

  private static double t1, t2;

  private DisplayMeanShift() {
    initialize();
    this.setTitle("Canopy Clusters (> 1.5% of population)");
  }

  @Override
  public void paint(Graphics g) {
    Graphics2D g2 = (Graphics2D) g;
    double sx = (double) res / ds;
    g2.setTransform(AffineTransform.getScaleInstance(sx, sx));

    // plot the axes
    g2.setColor(Color.BLACK);
    Vector dv = new DenseVector(2).assign(size / 2.0);
    Vector dv1 = new DenseVector(2).assign(t1);
    Vector dv2 = new DenseVector(2).assign(t2);
    DisplayDirichlet.plotRectangle(g2, new DenseVector(2).assign(2), dv);
    DisplayDirichlet.plotRectangle(g2, new DenseVector(2).assign(-2), dv);

    // plot the sample data
    g2.setColor(Color.DARK_GRAY);
    dv.assign(0.03);
    for (VectorWritable v : sampleData) {
      DisplayDirichlet.plotRectangle(g2, v.get(), dv);
    }
    int i = 0;
    for (MeanShiftCanopy canopy : canopies) {
      if (canopy.getBoundPoints().toList().size() > 0.015 * DisplayDirichlet.sampleData.size()) {
        g2.setColor(colors[Math.min(i++, DisplayDirichlet.colors.length - 1)]);
        for (int v : canopy.getBoundPoints().elements()) {
          DisplayDirichlet.plotRectangle(g2, sampleData.get(v).get(), dv);
        }
        DisplayDirichlet.plotEllipse(g2, canopy.getCenter(), dv1);
        DisplayDirichlet.plotEllipse(g2, canopy.getCenter(), dv2);
      }
    }
  }

  public static void main(String[] args) {
    RandomUtils.useTestSeed();
    DisplayDirichlet.generateSamples();
    List<Vector> points = new ArrayList<Vector>();
    for (VectorWritable sample : sampleData) {
      points.add(sample.get());
    }
    t1 = 1.5;
    t2 = 0.5;
    canopies = MeanShiftCanopyClusterer.clusterPoints(points, new EuclideanDistanceMeasure(), 0.005, t1, t2, 20);
    for (MeanShiftCanopy canopy : canopies) {
      log.info(canopy.toString());
    }
    new DisplayMeanShift();
  }

  static void generateResults() {
    DisplayDirichlet.generateResults(new NormalModelDistribution());
  }
}
