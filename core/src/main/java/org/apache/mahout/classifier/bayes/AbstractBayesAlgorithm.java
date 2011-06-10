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

import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.classifier.ClassifierResult;

import java.util.Collection;

abstract class AbstractBayesAlgorithm implements Algorithm {

  @Override
  public ClassifierResult[] classifyDocument(String[] document, Datastore datastore, String defaultCategory,
      int numResults) throws InvalidDatastoreException {
    Collection<String> categories = datastore.getKeys("labelWeight");
    TopK<ClassifierResult> topResults =
        new TopK<ClassifierResult>(numResults, ClassifierResult.COMPARE_BY_SCORE_AND_LABEL);
    for (String category : categories) {
      double prob = documentWeight(datastore, category, document);
      if (prob > 0.0) {
        topResults.offer(new ClassifierResult(category, prob));
      }
    }
    if (topResults.isEmpty()) {
      return new ClassifierResult[] { new ClassifierResult(defaultCategory, 0.0) };
    } else {
      return topResults.retrieve().toArray(new ClassifierResult[topResults.size()]);
    }
  }

  @Override
  public void initialize(Datastore datastore) throws InvalidDatastoreException {
    datastore.getKeys("labelWeight");
  }

  @Override
  public Collection<String> getLabels(Datastore datastore) throws InvalidDatastoreException {
    return datastore.getKeys("labelWeight");
  }
}
