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

import java.util.List;

import com.google.common.collect.Lists;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.dirichlet.models.DistanceMeasureClusterDistribution;
import org.apache.mahout.clustering.dirichlet.models.GaussianClusterDistribution;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

public final class TestDirichletClustering extends MahoutTestCase {

  private List<VectorWritable> sampleData;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    sampleData = Lists.newArrayList();
  }

  /**
   * Generate random samples and add them to the sampleData
   *
   * @param num int number of samples to generate
   * @param mx  double x-value of the sample mean
   * @param my  double y-value of the sample mean
   * @param sd  double standard deviation of the samples
   * @param card int cardinality of the generated sample vectors
   */
  private void generateSamples(int num, double mx, double my, double sd, int card) {
    System.out.println("Generating " + num + " samples m=[" + mx + ", " + my + "] sd=" + sd);
    for (int i = 0; i < num; i++) {
      DenseVector v = new DenseVector(card);
      for (int j = 0; j < card; j++) {
        v.set(j, UncommonDistributions.rNorm(mx, sd));
      }
      sampleData.add(new VectorWritable(v));
    }
  }

  /**
   * Generate 2-d samples for backwards compatibility with existing tests
   * @param num int number of samples to generate
   * @param mx  double x-value of the sample mean
   * @param my  double y-value of the sample mean
   * @param sd  double standard deviation of the samples
   */
  private void generateSamples(int num, double mx, double my, double sd) {
    generateSamples(num, mx, my, sd, 2);
  }

  private static void printResults(Iterable<Cluster[]> result, int significant) {
    int row = 0;
    for (Cluster[] r : result) {
      System.out.print("sample[" + row++ + "]= ");
      for (Cluster model : r) {
        if (model.count() > significant) {
          System.out.print(model.asFormatString(null) + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }

  @Test
  public void testDirichletCluster100() {
    System.out.println("testDirichletCluster100");
    generateSamples(40, 1, 1, 3);
    generateSamples(30, 1, 0, 0.1);
    generateSamples(30, 0, 1, 0.1);

    DirichletClusterer dc = new DirichletClusterer(sampleData,
                                                   new GaussianClusterDistribution(new VectorWritable(new DenseVector(2))),
                                                   1.0,
                                                   10,
                                                   1,
                                                   0);
    List<Cluster[]> result = dc.cluster(30);
    printResults(result, 2);
    assertNotNull(result);
  }

  @Test
  public void testDirichletGaussianCluster100() {
    System.out.println("testDirichletGaussianCluster100");
    generateSamples(40, 1, 1, 3);
    generateSamples(30, 1, 0, 0.1);
    generateSamples(30, 0, 1, 0.1);

    DirichletClusterer dc = new DirichletClusterer(sampleData,
                                                   new GaussianClusterDistribution(new VectorWritable(new DenseVector(2))),
                                                   1.0,
                                                   10,
                                                   1,
                                                   0);
    List<Cluster[]> result = dc.cluster(30);
    printResults(result, 2);
    assertNotNull(result);
  }

  @Test  
  public void testDirichletDMCluster100() {
    System.out.println("testDirichletDMCluster100");
    generateSamples(40, 1, 1, 3);
    generateSamples(30, 1, 0, 0.1);
    generateSamples(30, 0, 1, 0.1);

    DirichletClusterer dc = new DirichletClusterer(sampleData,
                                                   new DistanceMeasureClusterDistribution(new VectorWritable(new DenseVector(2))),
                                                   1.0,
                                                   10,
                                                   1,
                                                   0);
    List<Cluster[]> result = dc.cluster(30);
    printResults(result, 2);
    assertNotNull(result);
  }

}
