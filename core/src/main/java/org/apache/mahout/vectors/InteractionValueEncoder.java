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

import org.apache.mahout.math.Vector;

public class InteractionValueEncoder extends FeatureVectorEncoder {

  protected static final int INTERACTION_VALUE_HASH_SEED_1 = 100;
  protected static final int INTERACTION_VALUE_HASH_SEED_2 = 200;
  private String name1;
  private String name2;

  protected InteractionValueEncoder(String name1, String name2) {
    super(name1 + ":" + name2, 2);
    this.name1 = name1;
    this.name2 = name2;
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm The original form of the first value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(String originalForm, double w, Vector data) {
    throw new UnsupportedOperationException("Must have two arguments to encode interaction");
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm1 The original form of the first value as a string.
   * @param originalForm2 The original form of the second value as a string.
   * @param data          The vector to which the value should be added.
   */
  public void addToVector(String originalForm1, String originalForm2, Vector data) {
    int probes = getProbes();
    String name = getName();
    for (int i = 0; i < probes; i++) {
      int n = hash(name1, originalForm1, name2, originalForm2, i, data.size());
      trace(String.format("%s:%s", originalForm1, originalForm2), n);
      data.set(n, data.get(n) + 1);
    }
  }

  /**
   * Converts a value into a form that would help a human understand the internals of how the
   * value is being interpreted.  For text-like things, this is likely to be a list of the terms
   * found with associated weights (if any).
   *
   * @param originalForm The original form of the value as a string.
   * @return A string that a human can read.
   */
  @Override
  public String asString(String originalForm) {
    return String.format("%s:%s", getName(), originalForm);
  }
}

