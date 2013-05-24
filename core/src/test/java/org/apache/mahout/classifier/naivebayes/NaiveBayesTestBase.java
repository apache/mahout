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

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

public abstract class NaiveBayesTestBase extends MahoutTestCase {
  
  private NaiveBayesModel model;
  
  @Override
  public void setUp() throws Exception {
    super.setUp();
    model = createNaiveBayesModel();
    model.validate();
  }
  
  protected NaiveBayesModel getModel() {
    return model;
  }
  
  protected static double complementaryNaiveBayesThetaWeight(int label,
                                                             Matrix weightMatrix,
                                                             Vector labelSum,
                                                             Vector featureSum) {
    double weight = 0.0;
    double alpha = 1.0;
    for (int i = 0; i < featureSum.size(); i++) {
      double score = weightMatrix.get(i, label);
      double lSum = labelSum.get(label);
      double fSum = featureSum.get(i);
      double totalSum = featureSum.zSum();
      double numerator = fSum - score + alpha;
      double denominator = totalSum - lSum + featureSum.size();
      weight += Math.log(numerator / denominator);
    }
    return weight;
  }
  
  protected static double naiveBayesThetaWeight(int label,
                                                Matrix weightMatrix,
                                                Vector labelSum,
                                                Vector featureSum) {
    double weight = 0.0;
    double alpha = 1.0;
    for (int feature = 0; feature < featureSum.size(); feature++) {
      double score = weightMatrix.get(feature, label);
      double lSum = labelSum.get(label);
      double numerator = score + alpha;
      double denominator = lSum + featureSum.size();
      weight += Math.log(numerator / denominator);
    }
    return weight;
  }

  protected static NaiveBayesModel createNaiveBayesModel() {
    double[][] matrix = {
        { 0.7, 0.1, 0.1, 0.3 },
        { 0.4, 0.4, 0.1, 0.1 },
        { 0.1, 0.0, 0.8, 0.1 },
        { 0.1, 0.1, 0.1, 0.7 } };

    double[] labelSumArray = { 1.2, 1.0, 1.0, 1.0 };
    double[] featureSumArray = { 1.3, 0.6, 1.1, 1.2 };
    
    DenseMatrix weightMatrix = new DenseMatrix(matrix);
    DenseVector labelSum = new DenseVector(labelSumArray);
    DenseVector featureSum = new DenseVector(featureSumArray);
    
    double[] thetaNormalizerSum = {
        naiveBayesThetaWeight(0, weightMatrix, labelSum, featureSum),
        naiveBayesThetaWeight(1, weightMatrix, labelSum, featureSum),
        naiveBayesThetaWeight(2, weightMatrix, labelSum, featureSum),
        naiveBayesThetaWeight(3, weightMatrix, labelSum, featureSum) };

    // now generate the model
    return new NaiveBayesModel(weightMatrix, featureSum, labelSum, new DenseVector(thetaNormalizerSum), 1.0f);
  }
  
  protected static NaiveBayesModel createComplementaryNaiveBayesModel() {
    double[][] matrix = {
        { 0.7, 0.1, 0.1, 0.3 },
        { 0.4, 0.4, 0.1, 0.1 },
        { 0.1, 0.0, 0.8, 0.1 },
        { 0.1, 0.1, 0.1, 0.7 } };

    double[] labelSumArray = { 1.2, 1.0, 1.0, 1.0 };
    double[] featureSumArray = { 1.3, 0.6, 1.1, 1.2 };
    
    DenseMatrix weightMatrix = new DenseMatrix(matrix);
    DenseVector labelSum = new DenseVector(labelSumArray);
    DenseVector featureSum = new DenseVector(featureSumArray);
    
    double[] thetaNormalizerSum = {
        complementaryNaiveBayesThetaWeight(0, weightMatrix, labelSum, featureSum),
        complementaryNaiveBayesThetaWeight(1, weightMatrix, labelSum, featureSum),
        complementaryNaiveBayesThetaWeight(2, weightMatrix, labelSum, featureSum),
        complementaryNaiveBayesThetaWeight(3, weightMatrix, labelSum, featureSum) };

    // now generate the model
    return new NaiveBayesModel(weightMatrix, featureSum, labelSum, new DenseVector(thetaNormalizerSum), 1.0f);
  }
  
  protected static int maxIndex(Vector instance) {
    int maxIndex = -1;
    double maxScore = Integer.MIN_VALUE;
    for (Element label : instance.all()) {
      if (label.get() >= maxScore) {
        maxIndex = label.index();
        maxScore = label.get();
      }
    }
    return maxIndex;
  }
}
