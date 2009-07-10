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

package org.apache.mahout.common;

import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

/** The Base Model Class. Currently there are some Bayes Model elements which have to be refactored out later. */
public abstract class Model {

  private static final Logger log = LoggerFactory.getLogger(Model.class);

  public static final double DEFAULT_PROBABILITY = 0.5;

  protected final List<Map<Integer, Double>> featureLabelWeights = new ArrayList<Map<Integer, Double>>();

  protected final Map<String, Integer> featureList = new FastMap<String, Integer>();

  protected final Map<String, Integer> labelList = new HashMap<String, Integer>();

  protected final List<Double> sumFeatureWeight = new ArrayList<Double>();

  protected final Map<Integer, Double> sumLabelWeight = new HashMap<Integer, Double>();

  protected final Map<Integer, Double> thetaNormalizer = new HashMap<Integer, Double>();

  protected double sigma_jSigma_k = 0.0;

  protected static final double alpha_i = 1.0; // alpha_i can be improved upon for increased smoothing

  protected abstract double featureWeight(Integer label, Integer feature);

  protected abstract double getWeight(Integer label, Integer feature);

  protected abstract double getWeightUnprocessed(Integer label, Integer feature);

  public abstract void initializeNormalizer();

  public abstract void generateModel();

  protected double getSumLabelWeight(Integer label) {
    return nullToZero(sumLabelWeight.get(label));
  }

  protected double getThetaNormalizer(Integer label) {
    return nullToZero(thetaNormalizer.get(label));
  }

  protected double getSumFeatureWeight(Integer feature) {
    return nullToZero(sumFeatureWeight.get(feature));
  }

  private static double nullToZero(Double value) {
    return value == null ? 0.0 : value;
  }

  protected Integer getLabel(String label) {
    if (!labelList.containsKey(label)) {
      Integer labelId = labelList.size();
      labelList.put(label, labelId);
      return labelId;
    }
    return labelList.get(label);
  }

  protected Integer getFeature(String feature) {
    if (!featureList.containsKey(feature)) {
      Integer featureId = featureList.size();
      featureList.put(feature, featureId);
      return featureId;
    }
    return featureList.get(feature);
  }

  protected void setWeight(String labelString, String featureString, Double weight) {
    Integer feature = getFeature(featureString);
    Integer label = getLabel(labelString);
    setWeight(label, feature, weight);
  }

  protected void setWeight(Integer label, Integer feature, Double weight) {
    if (featureLabelWeights.size() <= feature) {
      throw new IllegalStateException("This should not happen");
    }
    featureLabelWeights.get(feature).put(label, weight);
  }

  protected void setSumFeatureWeight(Integer feature, double sum) {
    if (sumFeatureWeight.size() != feature) {
      throw new IllegalStateException("This should not happen");
    }
    sumFeatureWeight.add(feature, sum);
  }

  protected void setSumLabelWeight(Integer label, double sum) {
    if (sumLabelWeight.size() != label) {
      throw new IllegalStateException("This should not happen");
    }
    sumLabelWeight.put(label, sum);
  }

  protected void setThetaNormalizer(Integer label, double sum) {
    thetaNormalizer.put(label, sum);
  }

  public void initializeWeightMatrix() {
    log.info("{}", featureList.size());

    for (int i = 0; i < featureList.size(); i++) {
      featureLabelWeights.add(new HashMap<Integer, Double>(1));
    }
  }

  public void setSigma_jSigma_k(double sigma_jSigma_k) {
    this.sigma_jSigma_k = sigma_jSigma_k;
  }

  public void loadFeatureWeight(String labelString, String featureString, double weight) {
    setWeight(labelString, featureString, weight);
  }

  public void setSumFeatureWeight(String feature, double sum) {
    setSumFeatureWeight(getFeature(feature), sum);
  }

  public void setSumLabelWeight(String label, double sum) {
    setSumLabelWeight(getLabel(label), sum);
  }

  public void setThetaNormalizer(String label, double sum) {
    setThetaNormalizer(getLabel(label), sum);
  }

  /**
   * Get the weighted probability of the feature.
   *
   * @param labelString   The label of the feature
   * @param featureString The feature to calc. the prob. for
   * @return The weighted probability
   */
  public double featureWeight(String labelString, String featureString) {
    if (featureList.containsKey(featureString) == false) {
      return 0.0;
    }
    Integer feature = getFeature(featureString);
    Integer label = getLabel(labelString);
    return featureWeight(label, feature);
  }

  public Collection<String> getLabels() {
    return labelList.keySet();
  }

  public static Map<String, List<String>> generateNGrams(String line, int gramSize) {
    Map<String, List<String>> returnDocument = new HashMap<String, List<String>>();

    StringTokenizer tokenizer = new StringTokenizer(line);
    List<String> tokens = new ArrayList<String>();
    String labelName = tokenizer.nextToken();
    List<String> previousN_1Grams = new ArrayList<String>();
    while (tokenizer.hasMoreTokens()) {

      String next_token = tokenizer.nextToken();
      if (previousN_1Grams.size() == gramSize) {
        previousN_1Grams.remove(0);
      }

      previousN_1Grams.add(next_token);

      StringBuilder gramBuilder = new StringBuilder();

      for (String gram : previousN_1Grams) {
        gramBuilder.append(gram);
        String token = gramBuilder.toString();
        tokens.add(token);
        gramBuilder.append(' ');
      }
    }
    returnDocument.put(labelName, tokens);
    return returnDocument;
  }

  public static List<String> generateNGramsWithoutLabel(String line, int gramSize) {

    StringTokenizer tokenizer = new StringTokenizer(line);
    List<String> tokens = new ArrayList<String>();

    List<String> previousN_1Grams = new ArrayList<String>();
    while (tokenizer.hasMoreTokens()) {

      String next_token = tokenizer.nextToken();
      if (previousN_1Grams.size() == gramSize) {
        previousN_1Grams.remove(0);
      }

      previousN_1Grams.add(next_token);

      StringBuilder gramBuilder = new StringBuilder();

      for (String gram : previousN_1Grams) {
        gramBuilder.append(gram);
        String token = gramBuilder.toString();
        tokens.add(token);
        gramBuilder.append(' ');
      }
    }

    return tokens;
  }

}
