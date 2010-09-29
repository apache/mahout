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
package org.apache.mahout.clustering;

import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.dirichlet.UncommonDistributions;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestGaussianAccumulators extends MahoutTestCase {

  private static List<VectorWritable> sampleData = new ArrayList<VectorWritable>();

  private static final Logger log = LoggerFactory.getLogger(TestGaussianAccumulators.class);

  @Before
  public void setUp() throws Exception {
    super.setUp();
    generateSamples();
  }

  /**
   * Generate random samples and add them to the sampleData
   * 
   * @param num
   *          int number of samples to generate
   * @param mx
   *          double x-value of the sample mean
   * @param my
   *          double y-value of the sample mean
   * @param sd
   *          double standard deviation of the samples
   * @throws Exception 
   */
  public static void generateSamples(int num, double mx, double my, double sd) throws Exception {
    log.info("Generating {} samples m=[{}, {}] sd={}", new Object[] { num, mx, my, sd });
    for (int i = 0; i < num; i++) {
      sampleData.add(new VectorWritable(new DenseVector(new double[] { UncommonDistributions.rNorm(mx, sd),
          UncommonDistributions.rNorm(my, sd) })));
    }
  }

  /**
   * Generate random samples and add them to the sampleData
   * 
   * @param num
   *          int number of samples to generate
   * @param mx
   *          double x-value of the sample mean
   * @param my
   *          double y-value of the sample mean
   * @param sdx
   *          double x-value standard deviation of the samples
   * @param sdy
   *          double y-value standard deviation of the samples
   */
  public static void generate2dSamples(int num, double mx, double my, double sdx, double sdy) {
    log.info("Generating {} samples m=[{}, {}] sd=[{}, {}]", new Object[] { num, mx, my, sdx, sdy });
    for (int i = 0; i < num; i++) {
      sampleData.add(new VectorWritable(new DenseVector(new double[] { UncommonDistributions.rNorm(mx, sdx),
          UncommonDistributions.rNorm(my, sdy) })));
    }
  }

  private void generateSamples() throws Exception {
    generate2dSamples(500, 1, 2, 3, 4);
  }

  @Test
  public void testAccumulatorNoSamples() {
    GaussianAccumulator accumulator0 = new RunningSumsGaussianAccumulator();
    GaussianAccumulator accumulator1 = new OnlineGaussianAccumulator();
    accumulator0.compute();
    accumulator1.compute();
    assertEquals("N", accumulator0.getN(), accumulator1.getN(), EPSILON);
    assertEquals("Means", accumulator0.getMean(), accumulator1.getMean());
    assertEquals("Avg Stds", accumulator0.getAverageStd(), accumulator1.getAverageStd(), EPSILON);
  }

  @Test
  public void testAccumulatorResults() {
    GaussianAccumulator accumulator0 = new RunningSumsGaussianAccumulator();
    GaussianAccumulator accumulator1 = new OnlineGaussianAccumulator();
    for (VectorWritable vw : sampleData) {
      accumulator0.observe(vw.get(), 1);
      accumulator1.observe(vw.get(), 1);
    }
    accumulator0.compute();
    accumulator1.compute();
    assertEquals("N", accumulator0.getN(), accumulator1.getN(), EPSILON);
    assertEquals("Means", accumulator0.getMean().zSum(), accumulator1.getMean().zSum(), EPSILON);
    assertEquals("Stds", accumulator0.getStd().zSum(), accumulator1.getStd().zSum(), 0.01);
    //assertEquals("Variance", accumulator0.getVariance().zSum(), accumulator1.getVariance().zSum(), 1.6);
  }
}
