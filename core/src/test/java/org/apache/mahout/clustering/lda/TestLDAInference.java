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

package org.apache.mahout.clustering.lda;

import java.util.Iterator;
import java.util.Random;

import junit.framework.TestCase;

import org.apache.commons.math.distribution.PoissonDistribution;
import org.apache.commons.math.distribution.PoissonDistributionImpl;
import org.apache.commons.math.MathException;

import org.apache.mahout.matrix.DenseMatrix;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Matrix;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.common.RandomUtils;

public class TestLDAInference extends TestCase {

  private static final int NUM_TOPICS = 20;

  private Random random;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
    random = RandomUtils.getRandom();
  }

  /**
   * Generate random document vector
   * @param numWords int number of words in the vocabulary
   * @param numWords E[count] for each word
   */
  private Vector generateRandomDoc(int numWords, double sparsity) throws MathException {
    Vector v = new DenseVector(numWords);
    PoissonDistribution dist = new PoissonDistributionImpl(sparsity);
    for (int i = 0; i < numWords; i++) {
      // random integer
      v.setQuick(i, dist.inverseCumulativeProbability(random.nextDouble()) + 1);
    }
    return v;
  }

  private LDAState generateRandomState(int numWords, int numTopics) {
    double topicSmoothing = 50.0 / numTopics; // whatever
    Matrix m = new DenseMatrix(numTopics, numWords);
    double[] logTotals = new double[numTopics];

    for (int k = 0; k < numTopics; ++k) {
      double total = 0.0; // total number of pseudo counts we made
      for (int w = 0; w < numWords; ++w) {
        // A small amount of random noise, minimized by having a floor.
        double pseudocount = random.nextDouble() + 1.0E-10;
        total += pseudocount;
        m.setQuick(k, w, Math.log(pseudocount));
      }

      logTotals[k] = Math.log(total);
    }

    double ll = Double.NEGATIVE_INFINITY;
    return new LDAState(numTopics, numWords, topicSmoothing, m, logTotals, ll);
  }


  private void runTest(int numWords, double sparsity, int numTests) throws MathException {
    LDAState state = generateRandomState(numWords, NUM_TOPICS);
    LDAInference lda = new LDAInference(state);
    for (int t = 0; t < numTests; ++t) {
      Vector v = generateRandomDoc(numWords, sparsity);
      LDAInference.InferredDocument doc = lda.infer(v);

      assertEquals("wordCounts", doc.getWordCounts(), v);
      assertNotNull("gamma", doc.getGamma());
      for (Iterator<Vector.Element> iter = v.iterateNonZero();
          iter.hasNext(); ) {
        int w = iter.next().index();
        for (int k = 0; k < NUM_TOPICS; ++k) {
          double logProb = doc.phi(k, w);
          assertTrue(k + " " + w + " logProb " + logProb, logProb <= 0.0); 
        }
      }
      assertTrue("log likelihood", doc.logLikelihood <= 1.0E-10);
    }
  }


  public void testLDAEasy() throws MathException {
    runTest(10, 1.0, 5); // 1 word per doc in expectation
  }

  public void testLDASparse() throws MathException {
    runTest(100, 0.4, 5); // 40 words per doc in expectation
  }

  public void testLDADense() throws MathException {
    runTest(100, 3.0, 5); // 300 words per doc in expectation
  }
}
