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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public class BayesModel extends Model {

  private static final Logger log = LoggerFactory.getLogger(BayesModel.class);

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
    
    float weight = (float) Math.log(numerator /denominator);
    result = (-1.0f * (weight));

    return result;
  }

  @Override
  protected float getWeightUnprocessed(Integer label, Integer feature) {
    float result;
    Map<Integer, Float> featureWeights = featureLabelWeights.get(feature);

    if (featureWeights.containsKey(label)) {
      result = featureWeights.get(label);
    } else {
      result = 0.0f;
    }
    return result;
  }

  @Override
  public void InitializeNormalizer() {
    float perLabelWeightSumNormalisationFactor = Float.MAX_VALUE;

    log.info("{}", thetaNormalizer);
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
    log.info("{}", thetaNormalizer);
  }

  @Override
  public void GenerateModel() {
      float vocabCount = featureList.size();

      float[] perLabelThetaNormalizer = new float[labelList.size()];

      float perLabelWeightSumNormalisationFactor = Float.MAX_VALUE;

      for (int feature = 0, maxFeatures = featureList.size(); feature < maxFeatures; feature++) {
        Integer featureInt = feature;
        for (int label = 0, maxLabels = labelList.size(); label < maxLabels; label++) {

          Integer labelInt = label;
          float D_ij = getWeightUnprocessed(labelInt, featureInt);
          float sumLabelWeight = getSumLabelWeight(labelInt);
          // TODO srowen says sigma_j is unused
          //float sigma_j = getSumFeatureWeight(featureInt);

          float numerator = D_ij + alpha_i;
          float denominator = sumLabelWeight + vocabCount;

          float weight = (float) Math.log(numerator / denominator);

          if (D_ij != 0)
            setWeight(labelInt, featureInt, weight);

          perLabelThetaNormalizer[label] += weight;

        }
      }
      log.info("Normalizing Weights");
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
        Integer featureInt = feature;
        for (int label = 0, maxLabels = labelList.size(); label < maxLabels; label++) {
          Integer labelInt = label;
          float W_ij = getWeightUnprocessed(labelInt, featureInt);
          if (W_ij == 0)
            continue;
          float Sigma_W_ij = perLabelThetaNormalizer[label];
          float normalizedWeight = -1.0f * (W_ij / Sigma_W_ij);
          setWeight(labelInt, featureInt, normalizedWeight);
        }
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
