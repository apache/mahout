package org.apache.mahout.clustering.meanshift;

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

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.dirichlet.DisplayDirichlet;
import org.apache.mahout.clustering.dirichlet.UncommonDistributions;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.EuclideanDistanceMeasure;

class DisplayMeanShift extends DisplayDirichlet {
  private DisplayMeanShift() {
    initialize();
    this.setTitle("Canopy Clusters (> 1.5% of population)");
  }

  private static final long serialVersionUID = 1L;

  private static List<MeanShiftCanopy> canopies = new ArrayList<MeanShiftCanopy>();

  private static final List<List<Vector>> iterationCenters = new ArrayList<List<Vector>>();

  @Override
  public void paint(Graphics g) {
    Graphics2D g2 = (Graphics2D) g;
    double sx = (double) res / ds;
    g2.setTransform(AffineTransform.getScaleInstance(sx, sx));

    // plot the axes
    g2.setColor(Color.BLACK);
    Vector dv = new DenseVector(2).assign(size / 2);
    Vector dv1 = new DenseVector(2).assign(MeanShiftCanopy.t1);
    Vector dv2 = new DenseVector(2).assign(MeanShiftCanopy.t2);
    plotRectangle(g2, new DenseVector(2).assign(2), dv);
    plotRectangle(g2, new DenseVector(2).assign(-2), dv);

    // plot the sample data
    g2.setColor(Color.DARK_GRAY);
    dv.assign(0.03);
    for (Vector v : sampleData)
      plotRectangle(g2, v, dv);
    int i = 0;
    for (MeanShiftCanopy canopy : canopies)
      if (canopy.getBoundPoints().size() > 0.015 * sampleData.size()) {
        g2.setColor(colors[Math.min(i++, colors.length - 1)]);
        for (Vector v : canopy.getBoundPoints())
          plotRectangle(g2, v, dv);
        plotEllipse(g2, canopy.getCenter(), dv1);
        plotEllipse(g2, canopy.getCenter(), dv2);
      }
  }

  private static void testReferenceImplementation() {
    MeanShiftCanopy.config(new EuclideanDistanceMeasure(), 1.0, 0.05, 0.5);
    // add all points to the canopies
    for (Vector aRaw : sampleData) {
      MeanShiftCanopy.mergeCanopy(new MeanShiftCanopy(aRaw), canopies);
    }
    boolean done = false;
    while (!done) {// shift canopies to their centroids
      done = true;
      List<MeanShiftCanopy> migratedCanopies = new ArrayList<MeanShiftCanopy>();
      List<Vector> centers = new ArrayList<Vector>();
      for (MeanShiftCanopy canopy : canopies) {
        centers.add(canopy.getCenter());
        done = canopy.shiftToMean() && done;
        MeanShiftCanopy.mergeCanopy(canopy, migratedCanopies);
      }
      iterationCenters.add(centers);
      canopies = migratedCanopies;
    }
  }

  public static void main(String[] args) {
    UncommonDistributions.init("Mahout=Hadoop+ML".getBytes());
    generateSamples();
    testReferenceImplementation();
    for (MeanShiftCanopy canopy : canopies)
      System.out.println(canopy.toString());
    new DisplayMeanShift();
  }

  static void generateResults() {
    DisplayDirichlet.generateResults(new NormalModelDistribution());
  }
}
