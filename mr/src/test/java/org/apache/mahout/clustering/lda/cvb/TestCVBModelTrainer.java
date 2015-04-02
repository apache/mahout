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
package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixUtils;
import org.apache.mahout.math.function.DoubleFunction;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public final class TestCVBModelTrainer extends MahoutTestCase {

  private static final double ETA = 0.1;
  private static final double ALPHA = 0.1;

  @Test
  public void testInMemoryCVB0() throws Exception {
    String[] terms = new String[26];
    for (int i=0; i<terms.length; i++) {
      terms[i] = String.valueOf((char) (i + 'a'));
    }
    int numGeneratingTopics = 3;
    int numTerms = 26;
    Matrix matrix = ClusteringTestUtils.randomStructuredModel(numGeneratingTopics, numTerms, new DoubleFunction() {
      @Override public double apply(double d) {
        return 1.0 / Math.pow(d + 1.0, 2);
      }
    });

    int numDocs = 100;
    int numSamples = 20;
    int numTopicsPerDoc = 1;

    Matrix sampledCorpus = ClusteringTestUtils.sampledCorpus(matrix, RandomUtils.getRandom(),
                                                             numDocs, numSamples, numTopicsPerDoc);

    List<Double> perplexities = Lists.newArrayList();
    int numTrials = 1;
    for (int numTestTopics = 1; numTestTopics < 2 * numGeneratingTopics; numTestTopics++) {
      double[] perps = new double[numTrials];
      for (int trial = 0; trial < numTrials; trial++) {
        InMemoryCollapsedVariationalBayes0 cvb =
          new InMemoryCollapsedVariationalBayes0(sampledCorpus, terms, numTestTopics, ALPHA, ETA, 2, 1, 0);
        cvb.setVerbose(true);
        perps[trial] = cvb.iterateUntilConvergence(0, 5, 0, 0.2);
        System.out.println(perps[trial]);
      }
      Arrays.sort(perps);
      System.out.println(Arrays.toString(perps));
      perplexities.add(perps[0]);
    }
    System.out.println(Joiner.on(",").join(perplexities));
  }

  @Test
  public void testRandomStructuredModelViaMR() throws Exception {
    int numGeneratingTopics = 3;
    int numTerms = 9;
    Matrix matrix = ClusteringTestUtils.randomStructuredModel(numGeneratingTopics, numTerms, new DoubleFunction() {
      @Override
      public double apply(double d) {
        return 1.0 / Math.pow(d + 1.0, 3);
      }
    });

    int numDocs = 500;
    int numSamples = 10;
    int numTopicsPerDoc = 1;

    Matrix sampledCorpus = ClusteringTestUtils.sampledCorpus(matrix, RandomUtils.getRandom(1234),
                                                             numDocs, numSamples, numTopicsPerDoc);

    Path sampleCorpusPath = getTestTempDirPath("corpus");
    Configuration configuration = getConfiguration();
    MatrixUtils.write(sampleCorpusPath, configuration, sampledCorpus);
    int numIterations = 5;
    List<Double> perplexities = Lists.newArrayList();
    int startTopic = numGeneratingTopics - 1;
    int numTestTopics = startTopic;
    while (numTestTopics < numGeneratingTopics + 2) {
      Path topicModelStateTempPath = getTestTempDirPath("topicTemp" + numTestTopics);
      Configuration conf = getConfiguration();
      CVB0Driver cvb0Driver = new CVB0Driver();
      cvb0Driver.run(conf, sampleCorpusPath, null, numTestTopics, numTerms,
          ALPHA, ETA, numIterations, 1, 0, null, null, topicModelStateTempPath, 1234, 0.2f, 2,
          1, 3, 1, false);
      perplexities.add(lowestPerplexity(conf, topicModelStateTempPath));
      numTestTopics++;
    }
    int bestTopic = -1;
    double lowestPerplexity = Double.MAX_VALUE;
    for (int t = 0; t < perplexities.size(); t++) {
      if (perplexities.get(t) < lowestPerplexity) {
        lowestPerplexity = perplexities.get(t);
        bestTopic = t + startTopic;
      }
    }
    assertEquals("The optimal number of topics is not that of the generating distribution", 4, bestTopic);
    System.out.println("Perplexities: " + Joiner.on(", ").join(perplexities));
  }

  private static double lowestPerplexity(Configuration conf, Path topicModelTemp)
      throws IOException {
    double lowest = Double.MAX_VALUE;
    double current;
    int iteration = 2;
    while (!Double.isNaN(current = CVB0Driver.readPerplexity(conf, topicModelTemp, iteration))) {
      lowest = Math.min(current, lowest);
      iteration++;
    }
    return lowest;
  }

}
