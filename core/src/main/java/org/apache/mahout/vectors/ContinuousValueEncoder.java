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

/**
 * Continuous values are stored in fixed randomized location in the feature vector.
 */
public class ContinuousValueEncoder extends FeatureVectorEncoder {

  public ContinuousValueEncoder(String name) {
    super(name);
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(String originalForm, double weight, Vector data) {
    int probes = getProbes();
    String name = getName();
    for (int i = 0; i < probes; i++) {
      int n = hash(name, CONTINUOUS_VALUE_HASH_SEED + i, data.size());
      trace(null, n);
      data.set(n, data.get(n) + weight * Double.parseDouble(originalForm));
    }
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
    return getName() + ':' + originalForm;
  }
}
