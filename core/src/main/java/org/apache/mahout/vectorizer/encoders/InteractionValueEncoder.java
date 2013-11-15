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

import java.util.Locale;

import org.apache.mahout.math.Vector;

import com.google.common.base.Charsets;

public class InteractionValueEncoder extends FeatureVectorEncoder {
  private final FeatureVectorEncoder firstEncoder;
  private final FeatureVectorEncoder secondEncoder;

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
    throw new UnsupportedOperationException("addToVector is not supported for InteractionVectorEncoder");
  }

  /**
   * Adds a value to a vector. (Unsupported)
   *
   * @param originalForm The original form of the first value as a byte array.
   * @param data          The vector to which the value should be added.
   */
  @Override
  public void addToVector(byte[] originalForm, double w, Vector data) {
    throw new UnsupportedOperationException("addToVector is not supported for InteractionVectorEncoder");
  }

  /**
   * Adds a value to a vector.
   *
   * @param original1 The original form of the first value as a string.
   * @param original2 The original form of the second value as a string.
   * @param weight        How much to weight this interaction
   * @param data          The vector to which the value should be added.
   */
  public void addInteractionToVector(String original1, String original2, double weight, Vector data) {
    byte[] originalForm1 = bytesForString(original1);
    byte[] originalForm2 = bytesForString(original2);
    addInteractionToVector(originalForm1, originalForm2, weight, data);
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm1 The original form of the first value as a byte array.
   * @param originalForm2 The original form of the second value as a byte array.
   * @param weight        How much to weight this interaction
   * @param data          The vector to which the value should be added.
   */
  public void addInteractionToVector(byte[] originalForm1, byte[] originalForm2, double weight, Vector data) {
    String name = getName();
    double w = getWeight(originalForm1, originalForm2, weight);
    for (int i = 0; i < probes(); i++) {
      Iterable<Integer> jValues =
          secondEncoder.hashesForProbe(originalForm2, data.size(), name, i % secondEncoder.getProbes());
      for (Integer k : firstEncoder.hashesForProbe(originalForm1, data.size(), name, i % firstEncoder.getProbes())) {
        for (Integer j : jValues) {
          int n = (k + j) % data.size();
          if (isTraceEnabled()) {
            trace(String.format("%s:%s", new String(originalForm1, Charsets.UTF_8), new String(originalForm2,
		Charsets.UTF_8)), n);
          }
          data.set(n, data.get(n) + w);
        }
      }
    }
  }

  private int probes() {
    return getProbes();
  }

  protected double getWeight(byte[] originalForm1, byte[] originalForm2, double w) {
    return firstEncoder.getWeight(originalForm1, 1.0) * secondEncoder.getWeight(originalForm2, 1.0) * w;
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
    return String.format(Locale.ENGLISH, "%s:%s", getName(), originalForm);
  }

  @Override
  protected int hashForProbe(byte[] originalForm, int dataSize, String name, int probe) {
    return hash(name, probe, dataSize);
  }
}


