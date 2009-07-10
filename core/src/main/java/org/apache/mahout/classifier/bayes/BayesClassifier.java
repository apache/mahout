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

package org.apache.mahout.classifier.bayes;

import org.apache.hadoop.util.PriorityQueue;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.common.Classifier;
import org.apache.mahout.common.Model;

import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/** Classifies documents based on a {@link BayesModel}}. */
public class BayesClassifier implements Classifier {

  /**
   * Classify the document and return the top <code>numResults</code>
   *
   * @param model           The model
   * @param document        The document to classify
   * @param defaultCategory The default category to assign
   * @param numResults      The maximum number of results to return, ranked by score. Ties are broken by comparing the
   *                        category
   * @return A Collection of {@link ClassifierResult}s.
   */
  @Override
  public Collection<ClassifierResult> classify(Model model, String[] document, String defaultCategory, int numResults) {
    Collection<String> categories = model.getLabels();

    PriorityQueue<ClassifierResult> pq = new ClassifierResultPriorityQueue(numResults);
    ClassifierResult tmp;
    for (String category : categories) {
      double prob = documentWeight(model, category, document);
      if (prob > 0.0) {
        tmp = new ClassifierResult(category, prob);
        pq.insert(tmp);
      }
    }

    Deque<ClassifierResult> result = new LinkedList<ClassifierResult>();
    while ((tmp = pq.pop()) != null) {
      result.addLast(tmp);
    }
    if (result.isEmpty()) {
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
  @Override
  public ClassifierResult classify(Model model, String[] document, String defaultCategory) {
    ClassifierResult result = new ClassifierResult(defaultCategory);
    double max = Double.MAX_VALUE;
    Collection<String> categories = model.getLabels();

    for (String category : categories) {
      double prob = documentWeight(model, category, document);
      if (prob < max) {
        max = prob;
        result.setLabel(category);
      }
    }
    result.setScore(max);
    return result;
  }

  /**
   * Calculate the document weight as the multiplication of the {@link Model#featureWeight(String,
   * String)} for each word given the label
   *
   * @param model    The {@link Model}
   * @param label    The label to calculate the probability of
   * @param document The document
   * @return The probability
   * @see Model#featureWeight(String, String)
   */
  @Override
  public double documentWeight(Model model, String label, String[] document) {
    Map<String, int[]> wordList = new HashMap<String, int[]>(1000);
    for (String word : document) {
      int[] count = wordList.get(word);
      if (count == null) {
        count = new int[]{0};
        wordList.put(word, count);
      }
      count[0]++;
    }
    double result = 0.0;
    for (Map.Entry<String, int[]> entry : wordList.entrySet()) {
      String word = entry.getKey();
      int count = entry.getValue()[0];
      result += count * model.featureWeight(label, word);
    }
    return result;
  }

  private static class ClassifierResultPriorityQueue extends PriorityQueue<ClassifierResult> {

    private ClassifierResultPriorityQueue(int numResults) {
      initialize(numResults);
    }

    @Override
    protected boolean lessThan(Object a, Object b) {
      ClassifierResult cr1 = (ClassifierResult) a;
      ClassifierResult cr2 = (ClassifierResult) b;

      double score1 = cr1.getScore();
      double score2 = cr2.getScore();
      return score1 == score2 ? cr1.getLabel().compareTo(cr2.getLabel()) < 0 : score1 < score2;
    }
  }
}
