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

package org.apache.mahout.classifier.naivebayes.training;

import com.google.common.base.Preconditions;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.math.Vector;

public class ComplementaryThetaTrainer {

  private final Vector weightsPerFeature;
  private final Vector weightsPerLabel;
  private final Vector perLabelThetaNormalizer;
  private final double alphaI;
  private final double totalWeightSum;
  private final double numFeatures;

  public ComplementaryThetaTrainer(Vector weightsPerFeature, Vector weightsPerLabel, double alphaI) {
    Preconditions.checkNotNull(weightsPerFeature);
    Preconditions.checkNotNull(weightsPerLabel);
    this.weightsPerFeature = weightsPerFeature;
    this.weightsPerLabel = weightsPerLabel;
    this.alphaI = alphaI;
    perLabelThetaNormalizer = weightsPerLabel.like();
    totalWeightSum = weightsPerLabel.zSum();
    numFeatures = weightsPerFeature.getNumNondefaultElements();
  }

  public void train(int label, Vector perLabelWeight) {
    double labelWeight = labelWeight(label);
    // sum weights for each label including those with zero word counts
    for(int i = 0; i < perLabelWeight.size(); i++){
      Vector.Element perLabelWeightElement = perLabelWeight.getElement(i);
      updatePerLabelThetaNormalizer(label,
          ComplementaryNaiveBayesClassifier.computeWeight(featureWeight(perLabelWeightElement.index()),
              perLabelWeightElement.get(), totalWeightSum(), labelWeight, alphaI(), numFeatures()));
    }
  }

  protected double alphaI() {
    return alphaI;
  }

  protected double numFeatures() {
    return numFeatures;
  }

  protected double labelWeight(int label) {
    return weightsPerLabel.get(label);
  }

  protected double totalWeightSum() {
    return totalWeightSum;
  }

  protected double featureWeight(int feature) {
    return weightsPerFeature.get(feature);
  }

  // http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf - Section 3.2, Weight Magnitude Errors
  protected void updatePerLabelThetaNormalizer(int label, double weight) {
    perLabelThetaNormalizer.set(label, perLabelThetaNormalizer.get(label) + Math.abs(weight));
  }

  public Vector retrievePerLabelThetaNormalizer() {
    return perLabelThetaNormalizer.clone();
  }
}
