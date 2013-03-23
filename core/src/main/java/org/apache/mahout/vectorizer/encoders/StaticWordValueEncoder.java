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

import java.util.Collections;
import java.util.Map;

/**
 * Encodes a categorical values with an unbounded vocabulary.  Values are encoding by incrementing a
 * few locations in the output vector with a weight that is either defaulted to 1 or that is looked
 * up in a weight dictionary.  By default, only one probe is used which should be fine but could
 * cause a decrease in the speed of learning because more features will be non-zero. If a large
 * feature vector is used so that the probability of feature collisions is suitably small, then this
 * can be decreased to 1.  If a very small feature vector is used, the number of probes should
 * probably be increased to 3.
 */
public class StaticWordValueEncoder extends WordValueEncoder {
  private Map<String, Double> dictionary;
  private double missingValueWeight = 1;
  private final byte[] nameBytes;

  public StaticWordValueEncoder(String name) {
    super(name);
    nameBytes = bytesForString(name);
  }

  @Override
  protected int hashForProbe(byte[] originalForm, int dataSize, String name, int probe) {
    return hash(nameBytes, originalForm, WORD_LIKE_VALUE_HASH_SEED + probe, dataSize);
  }

  /**
   * Sets the weighting dictionary to be used by this encoder.  Also sets the missing value weight
   * to be half the smallest weight in the dictionary.
   *
   * @param dictionary The dictionary to use to look up weights.
   */
  public void setDictionary(Map<String, Double> dictionary) {
    this.dictionary = dictionary;
    setMissingValueWeight(Collections.min(dictionary.values()) / 2);
  }

  /**
   * Sets the weight that is to be used for values that do not appear in the dictionary.
   *
   * @param missingValueWeight The default weight for missing values.
   */
  public void setMissingValueWeight(double missingValueWeight) {
    this.missingValueWeight = missingValueWeight;
  }

  @Override
  protected double weight(byte[] originalForm) {
    double weight = missingValueWeight;
    if (dictionary != null) {
      String s = new String(originalForm, Charsets.UTF_8);
      if (dictionary.containsKey(s)) {
        weight = dictionary.get(s);
      }
    }
    return weight;
  }
}
