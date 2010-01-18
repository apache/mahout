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

package org.apache.mahout.clustering.dirichlet;

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;

import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalDistribution;
import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalModel;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.VectorWritable;

class DisplayASNDirichlet extends DisplayDirichlet {
  DisplayASNDirichlet() {
    initialize();
    this
        .setTitle("Dirichlet Process Clusters - Asymmetric Sampled Normal Distribution (>"
            + (int) (significance * 100) + "% of population)");
  }

  @Override
  public void paint(Graphics g) {
    super.plotSampleData(g);
    Graphics2D g2 = (Graphics2D) g;

    Vector dv = new DenseVector(2);
    int i = result.size() - 1;
    for (Model<VectorWritable>[] models : result) {
      g2.setStroke(new BasicStroke(i == 0 ? 3 : 1));
      g2.setColor(colors[Math.min(colors.length - 1, i--)]);
      for (Model<VectorWritable> m : models) {
        AsymmetricSampledNormalModel mm = (AsymmetricSampledNormalModel) m;
        dv.assign(mm.getStdDev().times(3));
        if (isSignificant(mm))
          plotEllipse(g2, mm.getMean(), dv);
      }
    }
  }

  public static void main(String[] args) {
    RandomUtils.useTestSeed();
    generateSamples();
    generateResults();
    new DisplayASNDirichlet();
  }

  static void generateResults() {
    DisplayDirichlet.generateResults(new AsymmetricSampledNormalDistribution(new VectorWritable(new DenseVector(2))));
  }
}
