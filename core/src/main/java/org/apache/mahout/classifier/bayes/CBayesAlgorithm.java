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

import org.apache.commons.lang.mutable.MutableDouble;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.math.function.ObjectIntProcedure;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

/**
 * Class implementing the Complementary Naive Bayes Classifier Algorithm
 * 
 */
public class CBayesAlgorithm extends AbstractBayesAlgorithm {
  
  @Override
  public ClassifierResult classifyDocument(String[] document,
                                           Datastore datastore,
                                           String defaultCategory) throws InvalidDatastoreException {
    ClassifierResult result = new ClassifierResult(defaultCategory);
    double max = Double.MIN_VALUE;
    Collection<String> categories = datastore.getKeys("labelWeight");
    
    for (String category : categories) {
      double prob = documentWeight(datastore, category, document);
      if (max < prob) {
        max = prob;
        result.setLabel(category);
      }
    }
    result.setScore(max);
    return result;
  }
  
  @Override
  public double featureWeight(Datastore datastore, String label, String feature) throws InvalidDatastoreException {
    
    double result = datastore.getWeight("weight", feature, label);
    double vocabCount = datastore.getWeight("sumWeight", "vocabCount");
    
    double featureSum = datastore.getWeight("weight", feature, "sigma_j");
    double totalSum = datastore.getWeight("sumWeight", "sigma_jSigma_k");
    double labelSum = datastore.getWeight("labelWeight", label);
    
    double thetaNormalizer = datastore.getWeight("thetaNormalizer", label);
    
    double numerator = featureSum - result + datastore.getWeight("params", "alpha_i");
    double denominator = totalSum - labelSum + vocabCount;
    
    double weight = Math.log(numerator / denominator);
    
    return weight / thetaNormalizer;
  }

  @Override
  public double documentWeight(final Datastore datastore,
                               final String label,
                               String[] document) {
    OpenObjectIntHashMap<String> wordList = new OpenObjectIntHashMap<String>(document.length / 2);
    for (String word : document) {
      wordList.adjustOrPutValue(word, 1, 1);
    }
    final MutableDouble result = new MutableDouble(0.0);
    
    wordList.forEachPair(new ObjectIntProcedure<String>() {
      @Override
      public boolean apply(String word, int frequency) {
        try {
          result.add(frequency * featureWeight(datastore, label, word));
        } catch (InvalidDatastoreException e) {
          throw new IllegalStateException(e);
        }
        return true;
      }
    });
    return result.doubleValue();
  }

}
