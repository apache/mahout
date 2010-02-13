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

package org.apache.mahout.classifier.bayes.interfaces;

import java.util.Collection;

import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.bayes.exceptions.InvalidDatastoreException;

/**
 * The algorithm interface for implementing variations of bayes Algorithm
 * 
 */
public interface Algorithm {
  /**
   * Initialize the data store and verifies the data in it.
   * 
   * @param datastore
   * @throws InvalidDatastoreException
   */
  void initialize(Datastore datastore) throws InvalidDatastoreException;
  
  /**
   * Classify the document and return the Result
   * 
   * @param document
   *          The document to classify
   * @param datastore
   *          The data store(InMemory, HBase)
   * @param defaultCategory
   *          The default category to assign Ties are broken by comparing the category
   * @return A Collection of {@link org.apache.mahout.classifier.ClassifierResult}s.
   * @throws InvalidDatastoreException
   */
  ClassifierResult classifyDocument(String[] document,
                                    Datastore datastore,
                                    String defaultCategory) throws InvalidDatastoreException;
  
  /**
   * Classify the document and return the top <code>numResults</code>
   * 
   * @param document
   *          The document to classify
   * @param datastore
   *          The {@link org.apache.mahout.classifier.bayes.interfaces.Datastore} (InMemory, HBase)
   * @param defaultCategory
   *          The default category to assign
   * @param numResults
   *          The maximum number of results to return, ranked by score. Ties are broken by comparing the
   *          category
   * @return A Collection of {@link org.apache.mahout.classifier.ClassifierResult}s.
   * @throws InvalidDatastoreException
   */
  ClassifierResult[] classifyDocument(String[] document,
                                      Datastore datastore,
                                      String defaultCategory,
                                      int numResults) throws InvalidDatastoreException;
  
  /**
   * Get the weighted probability of the feature.
   * 
   * @param datastore
   *          The {@link org.apache.mahout.classifier.bayes.interfaces.Datastore} (InMemory, HBase)
   * @param label
   *          The label of the feature
   * @param feature
   *          The feature to calc. the prob. for
   * @return The weighted probability
   * @throws InvalidDatastoreException
   */
  double featureWeight(Datastore datastore, String label, String feature) throws InvalidDatastoreException;
  
  /**
   * Calculate the document weight as the dot product of document vector and the corresponding weight vector
   * of a particular class
   * 
   * @param datastore
   *          The {@link org.apache.mahout.classifier.bayes.interfaces.Datastore} (InMemory, HBase)
   * @param label
   *          The label to calculate the probability of
   * @param document
   *          The document
   * @return The probability
   * @throws InvalidDatastoreException
   * @see Algorithm#featureWeight(Datastore, String, String)
   */
  double documentWeight(Datastore datastore, String label, String[] document) throws InvalidDatastoreException;
  
  /**
   * Returns the labels in the given Model
   * 
   * @param datastore
   *          The {@link org.apache.mahout.classifier.bayes.interfaces.Datastore} (InMemory, HBase)
   * @throws InvalidDatastoreException
   * @return {@link Collection} of labels
   */
  Collection<String> getLabels(Datastore datastore) throws InvalidDatastoreException;
}
