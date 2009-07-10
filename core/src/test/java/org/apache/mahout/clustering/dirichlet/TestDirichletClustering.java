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

import junit.framework.TestCase;
import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalDistribution;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;

import java.util.ArrayList;
import java.util.List;

public class TestDirichletClustering extends TestCase {

  private List<Vector> sampleData;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    UncommonDistributions.init("Mahout=Hadoop+ML".getBytes());
    sampleData = new ArrayList<Vector>();
  }

  /**
   * Generate random samples and add them to the sampleData
   *
   * @param num int number of samples to generate
   * @param mx  double x-value of the sample mean
   * @param my  double y-value of the sample mean
   * @param sd  double standard deviation of the samples
   */
  private void generateSamples(int num, double mx, double my, double sd) {
    System.out.println("Generating " + num + " samples m=[" + mx + ", " + my
        + "] sd=" + sd);
    for (int i = 0; i < num; i++) {
      sampleData.add(new DenseVector(new double[]{
          UncommonDistributions.rNorm(mx, sd),
          UncommonDistributions.rNorm(my, sd)}));
    }
  }

  private static void printResults(List<Model<Vector>[]> result, int significant) {
    int row = 0;
    for (Model<Vector>[] r : result) {
      System.out.print("sample[" + row++ + "]= ");
      for (Model<Vector> model : r) {
        if (model.count() > significant) {
          System.out.print(model.toString() + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }

  public void testDirichletCluster100() {
    System.out.println("testDirichletCluster100");
    generateSamples(40, 1, 1, 3);
    generateSamples(30, 1, 0, 0.1);
    generateSamples(30, 0, 1, 0.1);

    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        new NormalModelDistribution(), 1.0, 10, 1, 0);
    List<Model<Vector>[]> result = dc.cluster(30);
    printResults(result, 2);
    assertNotNull(result);
  }

  public void testDirichletCluster100s() {
    System.out.println("testDirichletCluster100s");
    generateSamples(40, 1, 1, 3);
    generateSamples(30, 1, 0, 0.1);
    generateSamples(30, 0, 1, 0.1);

    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        new SampledNormalDistribution(), 1.0, 10, 1, 0);
    List<Model<Vector>[]> result = dc.cluster(30);
    printResults(result, 2);
    assertNotNull(result);
  }

  public void testDirichletCluster100as() {
    System.out.println("testDirichletCluster100as");
    generateSamples(40, 1, 1, 3);
    generateSamples(30, 1, 0, 0.1);
    generateSamples(30, 0, 1, 0.1);

    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        new AsymmetricSampledNormalDistribution(), 1.0, 10, 1, 0);
    List<Model<Vector>[]> result = dc.cluster(30);
    printResults(result, 2);
    assertNotNull(result);
  }

  public void testDirichletCluster1000() {
    System.out.println("testDirichletCluster1000");
    generateSamples(400, 1, 1, 3);
    generateSamples(300, 1, 0, 0.1);
    generateSamples(300, 0, 1, 0.1);

    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        new NormalModelDistribution(), 1.0, 10, 1, 0);
    List<Model<Vector>[]> result = dc.cluster(30);
    printResults(result, 20);
    assertNotNull(result);
  }

  public void testDirichletCluster1000s() {
    System.out.println("testDirichletCluster1000s");
    generateSamples(400, 1, 1, 3);
    generateSamples(300, 1, 0, 0.1);
    generateSamples(300, 0, 1, 0.1);

    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        new SampledNormalDistribution(), 1.0, 10, 1, 0);
    List<Model<Vector>[]> result = dc.cluster(30);
    printResults(result, 20);
    assertNotNull(result);
  }

  public void testDirichletCluster1000as() {
    System.out.println("testDirichletCluster1000as");
    generateSamples(400, 1, 1, 3);
    generateSamples(300, 1, 0, 0.1);
    generateSamples(300, 0, 1, 0.1);

    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        new AsymmetricSampledNormalDistribution(), 1.0, 10, 1, 0);
    List<Model<Vector>[]> result = dc.cluster(30);
    printResults(result, 20);
    assertNotNull(result);
  }

  public void testDirichletCluster10000() {
    System.out.println("testDirichletCluster10000");
    generateSamples(4000, 1, 1, 3);
    generateSamples(3000, 1, 0, 0.1);
    generateSamples(3000, 0, 1, 0.1);

    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        new NormalModelDistribution(), 1.0, 10, 1, 0);
    List<Model<Vector>[]> result = dc.cluster(30);
    printResults(result, 200);
    assertNotNull(result);
  }

  public void testDirichletCluster10000as() {
    System.out.println("testDirichletCluster10000as");
    generateSamples(4000, 1, 1, 3);
    generateSamples(3000, 1, 0, 0.1);
    generateSamples(3000, 0, 1, 0.1);

    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        new AsymmetricSampledNormalDistribution(), 1.0, 10, 1, 0);
    List<Model<Vector>[]> result = dc.cluster(30);
    printResults(result, 200);
    assertNotNull(result);
  }

  public void testDirichletCluster10000s() {
    System.out.println("testDirichletCluster10000s");
    generateSamples(4000, 1, 1, 3);
    generateSamples(3000, 1, 0, 0.1);
    generateSamples(3000, 0, 1, 0.1);

    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        new SampledNormalDistribution(), 1.0, 10, 1, 0);
    List<Model<Vector>[]> result = dc.cluster(30);
    printResults(result, 200);
    assertNotNull(result);
  }
}
