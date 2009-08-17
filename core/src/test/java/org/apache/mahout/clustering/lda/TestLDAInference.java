package org.apache.mahout.clustering.lda;

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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import junit.framework.TestCase;

import org.apache.commons.math.distribution.PoissonDistribution;
import org.apache.commons.math.distribution.PoissonDistributionImpl;

import org.apache.mahout.matrix.DenseMatrix;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Matrix;
import org.apache.mahout.matrix.Vector;

public class TestLDAInference extends TestCase {

  private Random random;

  private static int NUM_TOPICS = 20;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    random = new Random();
  }

  /**
   * Generate random document vector
   * @param numWords int number of words in the vocabulary
   * @param numWords E[count] for each word
   */
  private Vector generateRandomDoc(int numWords, double sparsity) {
    Vector v = new DenseVector(numWords);
    try {
      PoissonDistribution dist = new PoissonDistributionImpl(sparsity);
      for (int i = 0; i < numWords; i++) {
        // random integer
        v.setQuick(i, dist.inverseCumulativeProbability(random.nextDouble()) + 1);
      }
    } catch (Exception e) {
      e.printStackTrace();
      fail("Caught " + e.toString());
    }
    return v;
  }

  private LDAState generateRandomState(int numWords, int numTopics) {
    double topicSmoothing = 50.0 / numTopics; // whatever
    Matrix m = new DenseMatrix(numTopics, numWords);
    double[] logTotals = new double[numTopics];
    double ll = Double.NEGATIVE_INFINITY;

    for (int k = 0; k < numTopics; ++k) {
      double total = 0.0; // total number of pseudo counts we made
      for (int w = 0; w < numWords; ++w) {
        // A small amount of random noise, minimized by having a floor.
        double pseudocount = random.nextDouble() + 1E-10;
        total += pseudocount;
        m.setQuick(k, w, Math.log(pseudocount));
      }

      logTotals[k] = Math.log(total);
    }

    return new LDAState(numTopics, numWords, topicSmoothing, m, logTotals, ll);
  }


  private void runTest(int numWords, double sparsity, int numTests) {
    LDAState state = generateRandomState(numWords, NUM_TOPICS);
    LDAInference lda = new LDAInference(state);
    for (int t = 0; t < numTests; ++t) {
      Vector v = generateRandomDoc(numWords, sparsity);
      LDAInference.InferredDocument doc = lda.infer(v);

      assertEquals("wordCounts", doc.wordCounts, v);
      assertNotNull("gamma", doc.gamma);
      for (Iterator<Vector.Element> iter = v.iterateNonZero();
          iter.hasNext(); ) {
        int w = iter.next().index();
        for (int k = 0; k < NUM_TOPICS; ++k) {
          double logProb = doc.phi(k, w);
          assertTrue(k + " " + w + " logProb " + logProb, logProb <= 0.0); 
        }
      }
      assertTrue("log likelihood", doc.logLikelihood <= 1E-10);
    }
  }


  public void testLDAEasy() {
    runTest(10, 1, 5); // 1 word per doc in expectation
  }

  public void testLDASparse() {
    runTest(100, 0.4, 5); // 40 words per doc in expectation
  }

  public void testLDADense() {
    runTest(100, 3, 5); // 300 words per doc in expectation
  }
}
