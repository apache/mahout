/*
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

package org.apache.mahout.vectors;

import org.apache.mahout.common.iterator.ArrayIterator;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;

public class InteractionValueEncoder extends FeatureVectorEncoder {

  protected static final int INTERACTION_VALUE_HASH_SEED_1 = 100;
  protected static final int INTERACTION_VALUE_HASH_SEED_2 = 200;
  protected static FeatureVectorEncoder firstEncoder;
  protected static FeatureVectorEncoder secondEncoder;

  public InteractionValueEncoder(String name, FeatureVectorEncoder encoderOne, FeatureVectorEncoder encoderTwo) {
    super(name, 2);
    firstEncoder = encoderOne;
    secondEncoder = encoderTwo;
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm The original form of the first value as a string.
   * @param data          The vector to which the value should be added.
   */
  @Override
  public void addToVector(String originalForm, double w, Vector data) {
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm1 The original form of the first value as a string.
   * @param originalForm2 The original form of the second value as a string.
   * @param data          The vector to which the value should be added.
   */
  public void addInteractionToVector(String originalForm1, String originalForm2, double weight, Vector data) {
    String name = getName();
    double w = getWeight(originalForm1, originalForm2, weight);
    for (int i = 0; i < probes(); i++) {
      for(Integer k : firstEncoder.hashesForProbe(originalForm1, data, name, i)){
        for(Integer j : secondEncoder.hashesForProbe(originalForm2, data, name, i)){
          int n = linearDoubleHash(hash1(k,name,i,data),hash2(k,name,i,data),j,data.size());
          trace(String.format("%s:%s", originalForm1, originalForm2), n);
          data.set(n, data.get(n) + w);
        }
      }
    }
  }

  private int probes() {
    return getProbes();
  }

  protected double getWeight(String originalForm1, String originalForm2, double w) {
    return firstEncoder.getWeight(originalForm1, 1.0) * secondEncoder.getWeight(originalForm2, 1.0) * w;
  }

  private int linearDoubleHash(int h1, int h2, int j, int modulus){
   int n = (h1 + (j+1)*h2) % modulus;
   if(n < 0){
    n = n+modulus;
   }
   return n;
  }

  /**
   * Converts a value into a form that would help a human understand the internals of how the value
   * is being interpreted.  For text-like things, this is likely to be a list of the terms found with
   * associated weights (if any).
   *
   * @param originalForm The original form of the value as a string.
   * @return A string that a human can read.
   */
  @Override
  public String asString(String originalForm) {
    return String.format("%s:%s", getName(), originalForm);
  }

  @Override
  protected int hashForProbe(String originalForm, Vector data, String name, int i) {
    return hash(name, i, data.size());
  }

  protected int hash1(int value, String name, int i, Vector data){
    return hash(name, i+value+INTERACTION_VALUE_HASH_SEED_1, data.size());
  }

  protected int hash2(int value, String name, int i, Vector data){
    return hash(name, i+value+INTERACTION_VALUE_HASH_SEED_2, data.size());
  }
}


