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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.SparseNormalModelDistribution;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector.Element;
import org.junit.After;
import org.junit.Before;

public class TestSparseModelClustering extends MahoutTestCase {

  private List<VectorWritable> sampleData;

  Random random;

  @Before
  protected void setUp() throws Exception {
    super.setUp();
    random = RandomUtils.getRandom();
    sampleData = new ArrayList<VectorWritable>();
  }

  /**
   * Generate random samples and add them to the sampleData
   *
   * @param num int number of samples to generate
   * @param mx  double value of the sample mean
   * @param sd  double standard deviation of the samples
   * @param card int cardinality of the generated sample vectors
   * @param pNz double probability a sample element is non-zero
   */
  private void generateSamples(int num, double mx, double sd, int card, double pNz) {
    Vector sparse = new SequentialAccessSparseVector(card);
    for (int i = 0; i < card; i++)
      if (random.nextDouble() < pNz)
        sparse.set(i, mx);
    System.out.println("Generating " + num + printSampleParameters(sparse, sd) + " pNz=" + pNz);
    for (int i = 0; i < num; i++) {
      SequentialAccessSparseVector v = new SequentialAccessSparseVector(card);
      for (int j = 0; j < card; j++) {
        if (sparse.get(j) > 0.0)
          v.set(j, UncommonDistributions.rNorm(mx, sd));
      }
      sampleData.add(new VectorWritable(v));
    }
  }

  @After
  public void tearDown() throws Exception {
  }

  public void testDirichletCluster100s() {
    System.out.println("testDirichletCluster100s");
    generateSamples(40, 5, 3, 50, 0.1);
    generateSamples(30, 3, 1, 50, 0.1);
    generateSamples(30, 1, 0.1, 50, 0.1);

    DirichletClusterer<VectorWritable> dc = new DirichletClusterer<VectorWritable>(sampleData, new SparseNormalModelDistribution(
        sampleData.get(0)), 1.0, 10, 1, 0);
    List<Model<VectorWritable>[]> result = dc.cluster(10);
    printResults(result, 1);
    assertNotNull(result);
  }

  private static void printResults(List<Model<VectorWritable>[]> result, int significant) {
    int row = 0;
    for (Model<VectorWritable>[] r : result) {
      int sig = 0;
      for (Model<VectorWritable> model : r) {
        if (model.count() > significant) {
          sig++;
        }
      }
      System.out.print("sample[" + row++ + "] (" + sig + ")= ");
      for (Model<VectorWritable> model : r) {
        if (model.count() > significant) {
          System.out.print(model.toString() + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }

  private static String printSampleParameters(Vector v, double stdDev) {
    StringBuilder buf = new StringBuilder();
    buf.append(" m=[");
    int nextIx = 0;
    if (v != null) {
      // handle sparse Vectors gracefully, suppressing zero values
      for (Iterator<Element> nzElems = v.iterateNonZero(); nzElems.hasNext();) {
        Element elem = nzElems.next();
        if (elem.index() > nextIx)
          buf.append("..{").append(elem.index()).append("}=");
        buf.append(String.format("%.2f", v.get(elem.index()))).append(", ");
        nextIx = elem.index() + 1;
      }
    }
    buf.append("] sd=").append(String.format("%.2f", stdDev)).append('}');
    return buf.toString();

  }

}
