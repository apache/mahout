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

package org.apache.mahout.common;

import org.apache.mahout.classifier.ClassifierResult;

import java.util.Collection;

/** Classifies documents based on a {@link org.apache.mahout.common.Model}. */
public interface Classifier {

  /**
   * Classify the document and return the top <code>numResults</code>
   *
   * @param model           The model
   * @param document        The document to classify
   * @param defaultCategory The default category to assign
   * @param numResults      The maximum number of results to return, ranked by score. Ties are broken by comparing the
   *                        category
   * @return A Collection of {@link org.apache.mahout.classifier.ClassifierResult}s.
   */
  Collection<ClassifierResult> classify(Model model, String[] document, String defaultCategory, int numResults);


  /**
   * Classify the document according to the {@link org.apache.mahout.common.Model}
   *
   * @param model           The trained {@link org.apache.mahout.common.Model}
   * @param document        The document to classify
   * @param defaultCategory The default category to assign if one cannot be determined
   * @return The single best category
   */
  ClassifierResult classify(Model model, String[] document, String defaultCategory);

  /**
   * Calculate the document probability as the multiplication of the {@link org.apache.mahout.common.Model#featureWeight(String,
   * String)} for each word given the label
   *
   * @param model    The {@link org.apache.mahout.common.Model}
   * @param label    The label to calculate the probability of
   * @param document The document
   * @return The probability
   * @see Model#featureWeight (String, String)
   */
  double documentWeight(Model model, String label, String[] document);


}
