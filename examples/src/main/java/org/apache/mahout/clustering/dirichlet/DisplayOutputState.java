package org.apache.mahout.clustering.dirichlet;

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

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.NormalModel;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;

class DisplayOutputState extends DisplayDirichlet {
  DisplayOutputState() {
    initialize();
    this.setTitle("Dirichlet Process Clusters - Map/Reduce Results (>"
        + (int) (significance * 100) + "% of population)");
  }

  private static final long serialVersionUID = 1L;

  public void paint(Graphics g) {
    super.plotSampleData(g);
    Graphics2D g2 = (Graphics2D) g;

    Vector dv = new DenseVector(2);
    int i = result.size() - 1;
    for (Model<Vector>[] models : result) {
      g2.setStroke(new BasicStroke(i == 0 ? 3 : 1));
      g2.setColor(colors[Math.min(colors.length - 1, i--)]);
      for (Model<Vector> m : models) {
        NormalModel mm = (NormalModel) m;
        dv.assign(mm.sd * 3);
        if (isSignificant(mm))
          plotEllipse(g2, mm.mean, dv);
      }
    }
  }

  /**
   * Return the contents of the given file as a String
   * 
   * @param fileName
   *            the String name of the file
   * @return the String contents of the file
   * @throws IOException
   *             if there is an error
   */
  public static List<Vector> readFile(String fileName) throws IOException {
    BufferedReader r = new BufferedReader(new FileReader(fileName));
    try {
      List<Vector> results = new ArrayList<Vector>();
      String line;
      while ((line = r.readLine()) != null)
        results.add(AbstractVector.decodeVector(line));
      return results;
    } finally {
      r.close();
    }
  }

  private static void getSamples() throws IOException {
    File f = new File("input");
    for (File g : f.listFiles())
      sampleData.addAll(readFile(g.getCanonicalPath()));
  }

  private static void getResults() throws IOException {
    result = new ArrayList<Model<Vector>[]>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY,
        "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution");
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, Integer.toString(20));
    conf.set(DirichletDriver.ALPHA_0_KEY, Double.toString(1.0));
    File f = new File("output");
    for (File g : f.listFiles()) {
      conf.set(DirichletDriver.STATE_IN_KEY, g.getCanonicalPath());
      DirichletState<Vector> dirichletState = DirichletMapper
          .getDirichletState(conf);
      result.add(dirichletState.getModels());
    }
  }

  public static void main(String[] args) {
    UncommonDistributions.init("Mahout=Hadoop+ML".getBytes());
    try {
      getSamples();
      getResults();
    } catch (IOException e) {
      e.printStackTrace();
    }
    new DisplayOutputState();
  }

  static void generateResults() {
    DisplayDirichlet.generateResults(new NormalModelDistribution());
  }
}
