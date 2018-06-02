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

import org.apache.mahout.math.Vector;

import java.util.Locale;

/**
 * Encodes words as sparse vector updates to a Vector.  Weighting is defined by a
 * sub-class.
 */
public abstract class WordValueEncoder extends FeatureVectorEncoder {
  private final byte[] nameBytes;

  protected WordValueEncoder(String name) {
    super(name, 2);
    nameBytes = bytesForString(name);
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(byte[] originalForm, double w, Vector data) {
    int probes = getProbes();
    String name = getName();
    double weight = getWeight(originalForm, w);
    for (int i = 0; i < probes; i++) {
      int n = hashForProbe(originalForm, data.size(), name, i);
      if (isTraceEnabled()) {
        trace(originalForm, n);        
      }
      data.set(n, data.get(n) + weight);
    }
  }


  @Override
  protected double getWeight(byte[] originalForm, double w) {
    return w * weight(originalForm);
  }

  @Override
  protected int hashForProbe(byte[] originalForm, int dataSize, String name, int probe) {
    return hash(nameBytes, originalForm, WORD_LIKE_VALUE_HASH_SEED + probe, dataSize);
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
    return String.format(Locale.ENGLISH, "%s:%s:%.4f", getName(), originalForm, weight(bytesForString(originalForm)));
  }

  protected abstract double weight(byte[] originalForm);
}
