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

import org.apache.mahout.math.Matrix;

public class LDAState {

  private final int numTopics;
  private final int numWords;
  private final double topicSmoothing;
  private final Matrix topicWordProbabilities; // log p(w|t) for topic=1..nTopics
  private final double[] logTotals; // log \sum p(w|t) for topic=1..nTopics
  private double logLikelihood; // log \sum p(w|t) for topic=1..nTopics
  
  public LDAState(int numTopics,
                  int numWords,
                  double topicSmoothing,
                  Matrix topicWordProbabilities,
                  double[] logTotals,
                  double ll) {
    this.numWords = numWords;
    this.numTopics = numTopics;
    this.topicSmoothing = topicSmoothing;
    this.topicWordProbabilities = topicWordProbabilities;
    this.logTotals = logTotals;
    this.logLikelihood = ll;
  }
  
  public double logProbWordGivenTopic(int word, int topic) {
    double logProb = topicWordProbabilities.get(topic, word);
    return logProb == Double.NEGATIVE_INFINITY ? -100.0 : logProb - logTotals[topic];
  }

  public double getLogTotal(int topic) {
    return logTotals[topic];
  }

  public int getNumTopics() {
    return numTopics;
  }

  public int getNumWords() {
    return numWords;
  }

  public double getTopicSmoothing() {
    return topicSmoothing;
  }

  public double getLogLikelihood() {
    return logLikelihood;
  }

  public void updateLogProbGivenTopic(int word, int topic, double logProbGivenTopic) {
    topicWordProbabilities.set(topic, word, LDAUtil.logSum(logProbGivenTopic, topicWordProbabilities.getQuick(topic, word)));
  }

  public void updateLogTotals(int topic, double logTotal) {
    logTotals[topic] = LDAUtil.logSum(logTotals[topic], logTotal);
  }

  public void setLogLikelihood(double logLikelihood) {
    this.logLikelihood = logLikelihood;
  }

}
