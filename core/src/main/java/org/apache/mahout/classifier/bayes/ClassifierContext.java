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

import java.util.Collection;

import org.apache.mahout.classifier.ClassifierResult;

/**
 * The Classifier Wrapper used for choosing the {@link Algorithm} and {@link Datastore}
 * 
 */
public class ClassifierContext {
  
  private final Algorithm algorithm;
  private final Datastore datastore;
  
  public ClassifierContext(Algorithm algorithm, Datastore datastore) {
    this.algorithm = algorithm;
    this.datastore = datastore;
  }
  
  /**
   * Initializes the Context. Gets the necessary data and checks if the Datastore is valid
   * 
   * @throws InvalidDatastoreException
   */
  public void initialize() throws InvalidDatastoreException {
    datastore.initialize();
    algorithm.initialize(this.datastore);
  }
  
  /**
   * Classify the document and return the Result
   * 
   * @param document
   *          The document to classify
   * @param defaultCategory
   *          The default category to assign Ties are broken by comparing the category
   * @return A Collection of {@link org.apache.mahout.classifier.ClassifierResult}s.
   * @throws InvalidDatastoreException
   */
  public ClassifierResult classifyDocument(String[] document, String defaultCategory) throws InvalidDatastoreException {
    return algorithm.classifyDocument(document, datastore, defaultCategory);
  }
  
  /**
   * Classify the document and return the top {@code numResults}
   *
   * @param document
   *          The document to classify
   * @param defaultCategory
   *          The default category to assign
   * @param numResults
   *          The maximum number of results to return, ranked by score. Ties are broken by comparing the
   *          category
   * @return A Collection of {@link ClassifierResult}s.
   * @throws InvalidDatastoreException
   */
  public ClassifierResult[] classifyDocument(String[] document,
                                             String defaultCategory,
                                             int numResults) throws InvalidDatastoreException {
    return algorithm.classifyDocument(document, datastore, defaultCategory, numResults);
  }
  
  /**
   * Gets the labels in the given model
   * 
   * @return Collection of Labels
   * @throws InvalidDatastoreException
   */
  public Collection<String> getLabels() throws InvalidDatastoreException {
    return algorithm.getLabels(datastore);
  }
  
}
