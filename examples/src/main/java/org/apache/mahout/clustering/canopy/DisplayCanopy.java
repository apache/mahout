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

package org.apache.mahout.clustering.canopy;

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.dirichlet.DisplayDirichlet;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

class DisplayCanopy extends DisplayDirichlet {
  DisplayCanopy() {
    initialize();
    this.setTitle("Canopy Clusters (> 5% of population)");
  }
  
  private static List<Canopy> canopies;
  
  private static final double t1 = 3.0;
  
  private static final double t2 = 1.6;
  
  @Override
  public void paint(Graphics g) {
    super.plotSampleData(g);
    Graphics2D g2 = (Graphics2D) g;
    Vector dv = new DenseVector(2);
    for (Canopy canopy : canopies) {
      if (canopy.getNumPoints() > DisplayDirichlet.sampleData.size() * 0.05) {
        g2.setStroke(new BasicStroke(2));
        g2.setColor(colors[1]);
        dv.assign(t1);
        Vector center = canopy.computeCentroid();
        DisplayDirichlet.plotEllipse(g2, center, dv);
        g2.setStroke(new BasicStroke(3));
        g2.setColor(colors[0]);
        dv.assign(t2);
        DisplayDirichlet.plotEllipse(g2, center, dv);
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
    canopies = CanopyClusterer.createCanopies(points, new ManhattanDistanceMeasure(), t1, t2);
    CanopyClusterer.updateCentroids(canopies);
    new DisplayCanopy();
  }
  
  static void generateResults() {
    DisplayDirichlet.generateResults(new NormalModelDistribution());
  }
}
