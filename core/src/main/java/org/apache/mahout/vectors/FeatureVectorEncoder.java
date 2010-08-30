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

import com.google.common.collect.Sets;
import org.apache.mahout.math.Vector;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * General interface for objects that record features into a feature vector.
 * <p/>
 * By convention, sub-classes should provide a constructor that accepts just a field name as well as
 * setters to customize properties of the conversion such as adding tokenizers or a weight
 * dictionary.
 */
public abstract class FeatureVectorEncoder {

  protected static final int CONTINUOUS_VALUE_HASH_SEED = 1;
  protected static final int WORD_LIKE_VALUE_HASH_SEED = 100;

  private final String name;
  private int probes;

  private Map<String, Set<Integer>> traceDictionary;

  protected FeatureVectorEncoder(String name) {
    this(name, 1);
  }

  protected FeatureVectorEncoder(String name, int probes) {
    this.name = name;
    this.probes = probes;
  }

  /**
   * Adds a value expressed in string form to a vector.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  public void addToVector(String originalForm, Vector data) {
    addToVector(originalForm, 1.0, data);
  }

  /**
   * Adds a weighted value expressed in string form to a vector.  In some cases it is convenient to
   * use this method to encode continuous values using the weight as the value.  In such cases, the
   * string value should typically be set to null.
   *
   * @param originalForm The original form of the value as a string.
   * @param weight       The weight to be applied to this feature.
   * @param data         The vector to which the value should be added.
   */
  public abstract void addToVector(String originalForm, double weight, Vector data);

  protected abstract int hashForProbe(String originalForm, Vector data, String name, int i);

  protected Iterable<Integer> hashesForProbe(String originalForm, Vector data, String name, int i){
    List<Integer> hashes = new ArrayList<Integer>();
    hashes.add(hashForProbe(originalForm,data,name,i));
  return hashes;
  }

  
  protected double getWeight(String originalFor, double w){
    return 1.0;
  }

  // ******* Utility functions used by most implementations

  /**
   * Hash a string and an integer into the range [0..numFeatures-1].
   *
   * @param term        The string.
   * @param probe       An integer that modifies the resulting hash.
   * @param numFeatures The range into which the resulting hash must fit.
   * @return An integer in the range [0..numFeatures-1] that has good spread for small changes in
   *         term and probe.
   */
  protected int hash(String term, int probe, int numFeatures) {
    long r = MurmurHash.hash64A(term.getBytes(Charset.forName("UTF-8")), probe) % numFeatures;
    if (r < 0) {
      r += numFeatures;
    }
    return (int) r;
  }

  /**
   * Hash two strings and an integer into the range [0..numFeatures-1].
   *
   * @param term1       The first string.
   * @param term2       The second string.
   * @param probe       An integer that modifies the resulting hash.
   * @param numFeatures The range into which the resulting hash must fit.
   * @return An integer in the range [0..numFeatures-1] that has good spread for small changes in
   *         term and probe.
   */
  protected int hash(String term1, String term2, int probe, int numFeatures) {
    long r = MurmurHash.hash64A(term1.getBytes(Charset.forName("UTF-8")), probe);
    r = MurmurHash.hash64A(term2.getBytes(Charset.forName("UTF-8")), (int) r) % numFeatures;
    if (r < 0) {
      r += numFeatures;
    }
    return (int) r;
  }

  /**
   * Hash four strings and an integer into the range [0..numFeatures-1].
   *
   * @param term1       The first string.
   * @param term2       The second string.
   * @param term3       The third string
   * @param term4       And the fourth.
   * @param probe       An integer that modifies the resulting hash.
   * @param numFeatures The range into which the resulting hash must fit.
   * @return An integer in the range [0..numFeatures-1] that has good spread for small changes in
   *         term and probe.
   */
  protected int hash(String term1, String term2, String term3, String term4, int probe, int numFeatures) {
    long r = MurmurHash.hash64A(term1.getBytes(Charset.forName("UTF-8")), probe);
    r = MurmurHash.hash64A(term2.getBytes(Charset.forName("UTF-8")), (int) r) % numFeatures;
    r = MurmurHash.hash64A(term3.getBytes(Charset.forName("UTF-8")), (int) r) % numFeatures;
    r = MurmurHash.hash64A(term4.getBytes(Charset.forName("UTF-8")), (int) r) % numFeatures;
    if (r < 0) {
      r += numFeatures;
    }
    return (int) r;
  }

  /**
   * Converts a value into a form that would help a human understand the internals of how the value
   * is being interpreted.  For text-like things, this is likely to be a list of the terms found
   * with associated weights (if any).
   *
   * @param originalForm The original form of the value as a string.
   * @return A string that a human can read.
   */
  public abstract String asString(String originalForm);

  public int getProbes() {
    return probes;
  }

  /**
   * Sets the number of locations in the feature vector that a value should be in.
   *
   * @param probes Number of locations to increment.
   */
  public void setProbes(int probes) {
    this.probes = probes;
  }

  public String getName() {
    return name;
  }

  protected void trace(String subName, int n) {
    if (traceDictionary != null) {
      String key = name;
      if (subName != null) {
        key = name + '=' + subName;
      }
      Set<Integer> trace = traceDictionary.get(key);
      if (trace == null) {
        trace = Sets.newHashSet(n);
        traceDictionary.put(key, trace);
      } else {
        trace.add(n);
      }
    }
  }

  public void setTraceDictionary(Map<String, Set<Integer>> traceDictionary) {
    this.traceDictionary = traceDictionary;
  }
}
