package org.apache.mahout.classifier.bayes;

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


import org.apache.mahout.common.Model;

import java.util.Map;


/**
 * 
 * 
 */
public class BayesModel extends Model {

  @Override
  protected float getWeight(Integer label, Integer feature) {
    float result = 0.0f;
    Map<Integer, Float> featureWeights = featureLabelWeights.get(feature);

    
    if (featureWeights.containsKey(label)) {
      result = featureWeights.get(label).floatValue();
    }
    
    float vocabCount = featureList.size();
    float sumLabelWeight = getSumLabelWeight(label);


    float numerator =  result + alpha_i;
    float denominator =(sumLabelWeight + vocabCount);
    
    float weight = new Double(Math.log(numerator /denominator)).floatValue();
    result = (-1.0f * (weight));

    return result;
  }

  @Override
  protected float getWeightUnprocessed(Integer label, Integer feature) {
    float result;
    Map<Integer, Float> featureWeights = featureLabelWeights.get(feature);

    if (featureWeights.containsKey(label)) {
      result = featureWeights.get(label).floatValue();
    } else {
      result = 0;
    }
    return result;
  }

  @Override
  public void InitializeNormalizer() {
    float perLabelWeightSumNormalisationFactor = Float.MAX_VALUE;

    System.out.println(thetaNormalizer);
    for (Integer label : thetaNormalizer.keySet()) {
      float Sigma_W_ij = thetaNormalizer.get(label);
      if (perLabelWeightSumNormalisationFactor > Math.abs(Sigma_W_ij)) {
        perLabelWeightSumNormalisationFactor = Math.abs(Sigma_W_ij);
      }
    }

    for (Integer label : thetaNormalizer.keySet()) {
      float Sigma_W_ij = thetaNormalizer.get(label);
      thetaNormalizer.put(label, Sigma_W_ij
          / perLabelWeightSumNormalisationFactor);
    }
    System.out.println(thetaNormalizer);
  }

  @Override
  public void GenerateModel() {
    try {
      float vocabCount = featureList.size();

      float[] perLabelThetaNormalizer = new float[labelList.size()];

      float perLabelWeightSumNormalisationFactor = Float.MAX_VALUE;

      for (int feature = 0, maxFeatures = featureList.size(); feature < maxFeatures; feature++) {
        for (int label = 0, maxLabels = labelList.size(); label < maxLabels; label++) {

          float D_ij = getWeightUnprocessed(label, feature);
          float sumLabelWeight = getSumLabelWeight(label);
          // TODO srowen says sigma_j is unused
          float sigma_j = getSumFeatureWeight(feature);

          float numerator = D_ij + alpha_i;
          float denominator = sumLabelWeight + vocabCount;

          Float weight = (float) Math.log(numerator / denominator);

          if (D_ij != 0)
            setWeight(label, feature, weight);

          perLabelThetaNormalizer[label] += weight;

        }
      }
      System.out.println("Normalizing Weights");
      for (int label = 0, maxLabels = labelList.size(); label < maxLabels; label++) {
        float Sigma_W_ij = perLabelThetaNormalizer[label];
        if (perLabelWeightSumNormalisationFactor > Math.abs(Sigma_W_ij)) {
          perLabelWeightSumNormalisationFactor = Math.abs(Sigma_W_ij);
        }
      }

      for (int label = 0, maxLabels = labelList.size(); label < maxLabels; label++) {
        float Sigma_W_ij = perLabelThetaNormalizer[label];
        perLabelThetaNormalizer[label] = Sigma_W_ij
            / perLabelWeightSumNormalisationFactor;
      }

      for (int feature = 0, maxFeatures = featureList.size(); feature < maxFeatures; feature++) {
        for (int label = 0, maxLabels = labelList.size(); label < maxLabels; label++) {
          float W_ij = getWeightUnprocessed(label, feature);
          if (W_ij == 0)
            continue;
          float Sigma_W_ij = perLabelThetaNormalizer[label];
          float normalizedWeight = -1.0f * (W_ij / Sigma_W_ij);
          setWeight(label, feature, normalizedWeight);
        }
      }
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

  
  /**
   * Get the weighted probability of the feature.
   * 
   * @param label The label of the feature
   * @param feature The feature to calc. the prob. for
   * @return The weighted probability
   */
  @Override
  public float FeatureWeight(Integer label, Integer feature) {
    return getWeight(label, feature);
  }

}
