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

package org.apache.mahout.vectorizer.encoders;

import com.google.common.base.Charsets;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import org.apache.mahout.math.Vector;

/**
 * Encodes words into vectors much as does WordValueEncoder while maintaining
 * an adaptive dictionary of values seen so far.  This allows weighting of terms
 * without a pre-scan of all of the data.
 */
public class AdaptiveWordValueEncoder extends WordValueEncoder {

  private final Multiset<String> dictionary;

  public AdaptiveWordValueEncoder(String name) {
    super(name);
    dictionary = HashMultiset.create();
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(String originalForm, double weight, Vector data) {
    dictionary.add(originalForm);
    super.addToVector(originalForm, weight, data);
  }

  @Override
  protected double getWeight(byte[] originalForm, double w) {
    return w * weight(originalForm);
  }

  @Override
  protected double weight(byte[] originalForm) {
    // the counts here are adjusted so that every observed value has an extra 0.5 count
    // as does a hypothetical unobserved value.  This smooths our estimates a bit and
    // allows the first word seen to have a non-zero weight of -log(1.5 / 2)
    double thisWord = dictionary.count(new String(originalForm, Charsets.UTF_8)) + 0.5;
    double allWords = dictionary.size() + dictionary.elementSet().size() * 0.5 + 0.5;
    return -Math.log(thisWord / allWords);
  }

  public Multiset<String> getDictionary() {
    return dictionary;
  }
}
