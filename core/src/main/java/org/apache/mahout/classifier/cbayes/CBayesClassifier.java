package org.apache.mahout.classifier.cbayes;
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

import org.apache.hadoop.util.PriorityQueue;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.common.Classifier;
import org.apache.mahout.common.Model;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/**
 * Classifies documents based on a {@link CBayesModel}.  
 */
public class CBayesClassifier implements Classifier{

  /**
   * Classify the document and return the top <code>numResults</code>
   *
   * @param model           The model
   * @param document        The document to classify
   * @param defaultCategory The default category to assign
   * @param numResults      The maximum number of results to return, ranked by score.  Ties are broken by comparing the category
   * @return A Collection of {@link org.apache.mahout.classifier.ClassifierResult}s.
   */
  public Collection<ClassifierResult> classify(Model model, String[] document, String defaultCategory, int numResults) {
    Collection<String> categories = model.getLabels();
    PriorityQueue pq = new ClassifierResultPriorityQueue(numResults);
    ClassifierResult tmp;
    for (String category : categories){
      float prob = documentProbability(model, category, document);
      if (prob < 0) {
        tmp = new ClassifierResult(category, prob);
        pq.insert(tmp);
      }
    }

    LinkedList<ClassifierResult> result = new LinkedList<ClassifierResult>();
    while ((tmp = (ClassifierResult) pq.pop()) != null) {
      result.addLast(tmp);
    }
    if (result.isEmpty()){
      result.add(new ClassifierResult(defaultCategory, 0));
    }
    return result;
  }

  /**
   * Classify the document according to the {@link org.apache.mahout.common.Model}
   *
   * @param model           The trained {@link org.apache.mahout.common.Model}
   * @param document        The document to classify
   * @param defaultCategory The default category to assign if one cannot be determined
   * @return The single best category
   */
  public ClassifierResult classify(Model model, String[] document, String defaultCategory) {
    ClassifierResult result = new ClassifierResult(defaultCategory);
    float min = 0.0f;
    Collection<String> categories = model.getLabels();

    for (String category : categories) {
      float prob = documentProbability(model, category, document);
      if (prob < min) {
        min = prob;
        result.setLabel(category);
      }
    }
    result.setScore(min);
    return result;
  }

  /**
   * Calculate the document probability as the multiplication of the {@link org.apache.mahout.common.Model#FeatureWeight(String, String)} for each word given the label
   *
   * @param model       The {@link org.apache.mahout.common.Model}
   * @param label       The label to calculate the probability of
   * @param document    The document
   * @return The probability
   * @see Model#FeatureWeight(String, String)
   */
  public float documentProbability(Model model, String label, String[] document) {
    float result = 0.0f;
    Map<String, Integer> wordList = new HashMap<String, Integer>(1000);
    for (String word : document) {
      if (wordList.containsKey(word)) {
        Integer count = wordList.get(word);
        count++;
        wordList.put(word, count);
      } else {
        wordList.put(word, 1);
      }      
    }
    for (Map.Entry<String, Integer> entry : wordList.entrySet()) {
      String word = entry.getKey();
      Integer count = entry.getValue();
      result += count * model.FeatureWeight(label, word);
    }
    return result;
  }

  
  private static class ClassifierResultPriorityQueue extends PriorityQueue {

    private ClassifierResultPriorityQueue(int numResults) {
      initialize(numResults);
    }

    protected boolean lessThan(Object a, Object b) {
      ClassifierResult cr1 = (ClassifierResult) a;
      ClassifierResult cr2 = (ClassifierResult) b;

      float score1 = cr1.getScore();
      float score2 = cr2.getScore();
      return score1 == score2 ? cr1.getLabel().compareTo(cr2.getLabel()) < 0 : score1 < score2;
    }
  }
}
