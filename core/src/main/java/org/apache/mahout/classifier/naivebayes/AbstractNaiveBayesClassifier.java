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

package org.apache.mahout.classifier.naivebayes;

import java.util.Iterator;

import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

/**
 * Class implementing the Naive Bayes Classifier Algorithm
 * 
 */
public abstract class AbstractNaiveBayesClassifier extends AbstractVectorClassifier {

  private final NaiveBayesModel model;
  
  protected AbstractNaiveBayesClassifier(NaiveBayesModel model) {
    this.model = model;
  }

  protected NaiveBayesModel getModel() {
    return model;
  }
  
  public abstract double getScoreForLabelFeature(int label, int feature);
  
  public double getScoreForLabelInstance(int label, Vector instance) {
    double result = 0.0;
    Iterator<Element> it = instance.iterateNonZero();
    while (it.hasNext()) {
      result +=  getScoreForLabelFeature(label, it.next().index());
    }
    return result;
  }
  
  @Override
  public int numCategories() {
    return model.getNumLabels();
  }

  @Override
  public Vector classify(Vector instance) {
    Vector score = model.getLabelSum().like();
    for (int i = 0; i < score.size(); i++) {
      score.set(i, getScoreForLabelInstance(i, instance));
    }
    return score;
  }

  @Override
  public double classifyScalar(Vector instance) {
    throw new UnsupportedOperationException("Not supported in Naive Bayes");
  }
  
}
