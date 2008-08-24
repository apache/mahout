package org.apache.mahout.common;

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

import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * The Base Model Class. Currently there are some Bayes Model elements which have to be refactored out later.
 * 
 */
public abstract class Model {

  private static final Logger log = LoggerFactory.getLogger(Model.class);

  public static final float DEFAULT_PROBABILITY = 0.5f;

  protected final List<Map<Integer, Float>> featureLabelWeights = new ArrayList<Map<Integer, Float>>();

  protected final Map<String, Integer> featureList = new FastMap<String, Integer>();

  protected final Map<String, Integer> labelList = new HashMap<String, Integer>();

  protected final List<Float> sumFeatureWeight = new ArrayList<Float>();

  protected final Map<Integer, Float> sumLabelWeight = new HashMap<Integer, Float>();

  protected final Map<Integer, Float> thetaNormalizer = new HashMap<Integer, Float>();

  protected Float sigma_jSigma_k = new Float(0);

  protected final Float alpha_i = 1.0f; // alpha_i can be improved upon for increased smoothing
  
  protected abstract float FeatureWeight(Integer label, Integer feature);
  
  protected abstract float getWeight(Integer label, Integer feature);

  protected abstract float getWeightUnprocessed(Integer label, Integer feature);
  
  public abstract void InitializeNormalizer();

  public abstract void GenerateModel();
  
  protected float getSumLabelWeight(Integer label) {
    float result = 0.0f;
    Float numSeen = sumLabelWeight.get(label);
    if (numSeen != null) {
      result = numSeen;
    }
    return result;
  }

  protected float getThetaNormalizer(Integer label) {
    float result = 0.0f;
    Float numSeen = thetaNormalizer.get(label);
    if (numSeen != null) {
      result = numSeen;
    }
    return result;
  }

  protected float getSumFeatureWeight(Integer feature) {
    float result = 0.0f;
    Float numSeen = sumFeatureWeight.get(feature);
    if (numSeen != null) {
      result = numSeen;
    }
    return result;
  }

  protected Integer getLabel(String label) {
    if (!labelList.containsKey(label)) {
      
      Integer labelId = Integer.valueOf(labelList.size());
      labelList.put(label, labelId);
    }
    return labelList.get(label);
  }

  protected Integer getFeature(String feature) {
    if (!featureList.containsKey(feature)) {
      
      Integer featureId = Integer.valueOf(featureList.size());
      featureList.put(feature, featureId);
    }
    return featureList.get(feature);
  }

  protected void setWeight(String labelString, String featureString, Float weight) {
    Integer feature = getFeature(featureString);
    Integer label = getLabel(labelString);
    setWeight(label, feature, weight);
  }

  protected void setWeight(Integer label, Integer feature, Float weight) {
    if (featureLabelWeights.size() <= feature) {
      throw new IllegalStateException("This should not happen");
    }
    featureLabelWeights.get(feature).put(label, new Float(weight));
  }

  protected void setSumFeatureWeight(Integer feature, float sum) {
    if (sumFeatureWeight.size() != feature)
      throw new IllegalStateException("This should not happen");
    sumFeatureWeight.add(feature, new Float(sum));
  }

  protected void setSumLabelWeight(Integer label, float sum) {
    if (sumLabelWeight.size() != label)
      throw new IllegalStateException("This should not happen");
    sumLabelWeight.put(label, new Float(sum));
  }

  protected void setThetaNormalizer(Integer label, float sum) {
    thetaNormalizer.put(label, new Float(sum));
  }

  public void initializeWeightMatrix() {
    log.info("{}", featureList.size());

    for (int i = 0; i < featureList.size(); i++)
      featureLabelWeights.add(new HashMap<Integer, Float>(1));
  }

  public void setSigma_jSigma_k(Float sigma_jSigma_k) {
    this.sigma_jSigma_k = sigma_jSigma_k;
  }

  public void loadFeatureWeight(String labelString, String featureString,
      float weight) {
    setWeight(labelString, featureString, weight);
  }

  public void setSumFeatureWeight(String feature, float sum) {
    setSumFeatureWeight(getFeature(feature), sum);
  }

  public void setSumLabelWeight(String label, float sum) {
    setSumLabelWeight(getLabel(label), sum);
  }

  public void setThetaNormalizer(String label, float sum) {
    setThetaNormalizer(getLabel(label), sum);
  }

  /**
   * Get the weighted probability of the feature.
   * 
   * @param labelString The label of the feature
   * @param featureString The feature to calc. the prob. for
   * @return The weighted probability
   */
  public float FeatureWeight(String labelString, String featureString) {
    if (featureList.containsKey(featureString) == false)
      return 0.0f;
    Integer feature = getFeature(featureString);
    Integer label = getLabel(labelString);
    return FeatureWeight(label, feature);
  }

  public Collection<String> getLabels() {
    return labelList.keySet();
  }
  
  public static Map<String, List<String>> generateNGrams(String line, int gramSize)
  {
    Map<String, List<String>> returnDocument = new HashMap<String, List<String>>();
    
    StringTokenizer tokenizer = new StringTokenizer(line);
    List<String> tokens = new ArrayList<String>();
    String labelName = tokenizer.nextToken();
    List<String> previousN_1Grams  = new ArrayList<String>();
    while (tokenizer.hasMoreTokens()) {
      
      String next_token = tokenizer.nextToken();
      if(previousN_1Grams.size() == gramSize)
        previousN_1Grams.remove(0);
   
      previousN_1Grams.add(next_token);
      
      StringBuilder gramBuilder = new StringBuilder();
     
      for(String gram: previousN_1Grams)
      {
        gramBuilder.append(gram);
        String token = gramBuilder.toString();        
        tokens.add(token);
        gramBuilder.append(" ");
      }
    }
    returnDocument.put(labelName, tokens);
    return returnDocument;
  }
  
  public static List<String> generateNGramsWithoutLabel(String line, int gramSize)
  {
  
    StringTokenizer tokenizer = new StringTokenizer(line);
    List<String> tokens = new ArrayList<String>();
   
    List<String> previousN_1Grams  = new ArrayList<String>();
    while (tokenizer.hasMoreTokens()) {
      
      String next_token = tokenizer.nextToken();
      if(previousN_1Grams.size() == gramSize)
        previousN_1Grams.remove(0);
   
      previousN_1Grams.add(next_token);
      
      StringBuilder gramBuilder = new StringBuilder();
     
      for(String gram: previousN_1Grams)
      {
        gramBuilder.append(gram);
        String token = gramBuilder.toString();        
        tokens.add(token);
        gramBuilder.append(" ");
      }
    }
    
    return tokens;
  }

}
