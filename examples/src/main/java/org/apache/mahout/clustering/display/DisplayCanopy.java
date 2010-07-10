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

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.canopy.CanopyClusterer;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

class DisplayCanopy extends DisplayClustering {

  private static final long serialVersionUID = 1L;

  private static List<Canopy> canopies;

  private static final double T1 = 3.0;

  private static final double T2 = 1.6;

  DisplayCanopy() {
    initialize();
    this.setTitle("Canopy Clusters (>" + (int) (SIGNIFICANCE * 100) + "% of population)");
  }

  @Override
  public void paint(Graphics g) {
    Graphics2D g2 = (Graphics2D) g;
    plotSampleData(g2);
    Vector dv = new DenseVector(2);
    for (Canopy canopy : canopies) {
      if (canopy.getNumPoints() > DisplayClustering.SAMPLE_DATA.size() * SIGNIFICANCE) {
        g2.setStroke(new BasicStroke(2));
        g2.setColor(COLORS[1]);
        dv.assign(T1);
        Vector center = canopy.computeCentroid();
        DisplayClustering.plotEllipse(g2, center, dv);
        g2.setStroke(new BasicStroke(3));
        g2.setColor(COLORS[0]);
        dv.assign(T2);
        DisplayClustering.plotEllipse(g2, center, dv);
      }
    }
  }

  public static void main(String[] args) {
    RandomUtils.useTestSeed();
    DisplayClustering.generateSamples();
    List<Vector> points = new ArrayList<Vector>();
    for (VectorWritable sample : SAMPLE_DATA) {
      points.add(sample.get());
    }
    canopies = CanopyClusterer.createCanopies(points, new ManhattanDistanceMeasure(), T1, T2);
    CanopyClusterer.updateCentroids(canopies);
    new DisplayCanopy();
  }
}
