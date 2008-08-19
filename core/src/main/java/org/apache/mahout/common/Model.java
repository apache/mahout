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


import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Collection;
import java.util.StringTokenizer;

/**
 * The Base Model Class. Currently there are some Bayes Model elements which have to be refactored out later.
 * 
 */
public abstract class Model {

  protected List<Map<Integer, Float>> featureLabelWeights = new ArrayList<Map<Integer, Float>>();

  protected Map<String, Integer> featureList = new FastMap<String, Integer>();

  protected Map<String, Integer> labelList = new HashMap<String, Integer>();

  protected List<Float> sumFeatureWeight = new ArrayList<Float>();

  protected Map<Integer, Float> sumLabelWeight = new HashMap<Integer, Float>();

  protected Map<Integer, Float> thetaNormalizer = new HashMap<Integer, Float>();

  protected Float sigma_jSigma_k = new Float(0);

  protected Float alpha_i = 1.0f; // alpha_i can be improved upon for increased smoothing
  
  public static float  DEFAULT_PROBABILITY = 0.5f;
  
  
  protected abstract float FeatureWeight(Integer label, Integer feature);
  
  protected abstract float getWeight(Integer label, Integer feature);

  protected abstract float getWeightUnprocessed(Integer label, Integer feature);
  
  public abstract void InitializeNormalizer();

  public abstract void GenerateModel();
  
  protected float getSumLabelWeight(Integer label) {
    float result = 0;
    Float numSeen = sumLabelWeight.get(label);
    if (numSeen != null) {
      result = ((float) numSeen);
    }
    return result;
  }

  protected float getThetaNormalizer(Integer label) {
    float result = 0.0f;
    Float numSeen = thetaNormalizer.get(label);
    if (numSeen != null) {
      result = ((float) numSeen);
    }
    return result;
  }

  protected float getSumFeatureWeight(Integer feature) {
    float result = 0;
    Float numSeen = sumFeatureWeight.get(feature);
    if (numSeen != null) {
      result = ((float) numSeen);
    }
    return result;
  }

  protected Integer getLabel(String label) {
    if (!labelList.containsKey(label)) {
      
      Integer labelId = Integer.valueOf(labelList.size());
      labelList.put(label, labelId);
    }
    Integer labelId = labelList.get(label);
    return labelId;
  }

  protected Integer getFeature(String feature) {
    if (!featureList.containsKey(feature)) {
      
      Integer featureId = Integer.valueOf(featureList.size());
      featureList.put(feature, featureId);
    }
    Integer featureId = featureList.get(feature);
    return featureId;
  }

  protected void setWeight(String labelString, String featureString, Float weight)
      throws Exception {
    Integer feature = getFeature(featureString);
    Integer label = getLabel(labelString);
    setWeight(label, feature, weight);
  }

  protected void setWeight(Integer label, Integer feature, Float weight) throws Exception {
    if (featureLabelWeights.size() <= feature) {
      // System.out.println(feature + "," + featureLabelWeights.size());
      // System.in.read();
      throw new Exception("This should not happen");

    }
    featureLabelWeights.get(feature).put(label, new Float(weight));
  }

  protected void setSumFeatureWeight(Integer feature, float sum) throws Exception {
    if (sumFeatureWeight.size() != feature)
      throw new Exception("This should not happen");
    sumFeatureWeight.add(feature, new Float(sum));
  }

  protected void setSumLabelWeight(Integer label, float sum) throws Exception {
    if (sumLabelWeight.size() != label)
      throw new Exception("This should not happen");
    sumLabelWeight.put(label, new Float(sum));
  }

  protected void setThetaNormalizer(Integer label, float sum) {
    thetaNormalizer.put(label, new Float(sum));
  }

  public void initializeWeightMatrix() {
    System.out.println(featureList.size());

    for (int i = 0; i < featureList.size(); i++)
      featureLabelWeights.add(new HashMap<Integer, Float>(1));
  }

  public void setSigma_jSigma_k(Float sigma_jSigma_k) {
    this.sigma_jSigma_k = sigma_jSigma_k;
  }

  public void loadFeatureWeight(String labelString, String featureString,
      float weight) {
    try {
      setWeight(labelString, featureString, weight);
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

  public void setSumFeatureWeight(String feature, float sum) {
    try {
      setSumFeatureWeight(getFeature(feature), sum);
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

  public void setSumLabelWeight(String label, float sum) {
    try {
      setSumLabelWeight(getLabel(label), sum);
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
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
