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
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixUtils;
import org.apache.mahout.math.function.DoubleFunction;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.apache.mahout.clustering.ClusteringTestUtils.randomStructuredModel;
import static org.apache.mahout.clustering.ClusteringTestUtils.sampledCorpus;

public class TestCVBModelTrainer extends MahoutTestCase {
  private double eta = 0.1;
  private double alpha = 0.1;

  @Test
  public void testInMemoryCVB0() throws Exception {
    int numGeneratingTopics = 5;
    int numTerms = 26;
    String[] terms = new String[26];
    for(int i=0; i<terms.length; i++) {
      terms[i] = "" + ((char)(i + 97));
    }
    Matrix matrix = randomStructuredModel(numGeneratingTopics, numTerms, new DoubleFunction() {
      @Override public double apply(double d) {
        return 1d / Math.pow(d+1, 2);
      }
    });

    int numDocs = 100;
    int numSamples = 20;
    int numTopicsPerDoc = 1;

    Matrix sampledCorpus = sampledCorpus(matrix, new Random(12345),
        numDocs, numSamples, numTopicsPerDoc);

    List<Double> perplexities = Lists.newArrayList();
    int numTrials = 2;
    for(int numTestTopics = 1; numTestTopics < 2 * numGeneratingTopics; numTestTopics++) {
      double[] perps = new double[numTrials];
      for(int trial = 0; trial < numTrials; trial++) {
        InMemoryCollapsedVariationalBayes0 cvb =
          new InMemoryCollapsedVariationalBayes0(sampledCorpus, terms, numTestTopics, alpha, eta,
              2, 1, 0, (trial+1) * 123456L);
        cvb.setVerbose(true);
        perps[trial] = cvb.iterateUntilConvergence(0, 20, 0, 0.2);
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
    Matrix matrix = randomStructuredModel(numGeneratingTopics, numTerms, new DoubleFunction() {
      @Override public double apply(double d) {
        return 1d / Math.pow(d+1, 3);
      }
    });

    int numDocs = 500;
    int numSamples = 10;
    int numTopicsPerDoc = 1;

    Matrix sampledCorpus = sampledCorpus(matrix, new Random(1234),
        numDocs, numSamples, numTopicsPerDoc);

    Path sampleCorpusPath = getTestTempDirPath("corpus");
    MatrixUtils.write(sampleCorpusPath, new Configuration(), sampledCorpus);
    int numIterations = 5;
    List<Double> perplexities = Lists.newArrayList();
    int startTopic = numGeneratingTopics - 1;
    int numTestTopics = startTopic;
    while(numTestTopics < numGeneratingTopics + 2) {
      CVB0Driver driver = new CVB0Driver();
      Path topicModelStateTempPath = getTestTempDirPath("topicTemp" + numTestTopics);
      Configuration conf = new Configuration();
      driver.run(conf, sampleCorpusPath, null, numTestTopics, numTerms,
          alpha, eta, numIterations, 1, 0, null, null, topicModelStateTempPath, 1234, 0.2f, 2,
          1, 10, 1, false);
      perplexities.add(lowestPerplexity(conf, topicModelStateTempPath));
      numTestTopics++;
    }
    int bestTopic = -1;
    double lowestPerplexity = Double.MAX_VALUE;
    for(int t = 0; t < perplexities.size(); t++) {
      if(perplexities.get(t) < lowestPerplexity) {
        lowestPerplexity = perplexities.get(t);
        bestTopic = t + startTopic;
      }
    }
    assertEquals("The optimal number of topics is not that of the generating distribution",
        bestTopic, numGeneratingTopics);
    System.out.println("Perplexities: " + Joiner.on(", ").join(perplexities));
  }

  private static double lowestPerplexity(Configuration conf, Path topicModelTemp)
      throws IOException {
    double lowest = Double.MAX_VALUE;
    double current;
    int iteration = 2;
    while(!Double.isNaN(current = CVB0Driver.readPerplexity(conf, topicModelTemp, iteration))) {
      lowest = Math.min(current, lowest);
      iteration++;
    }
    return lowest;
  }

}
