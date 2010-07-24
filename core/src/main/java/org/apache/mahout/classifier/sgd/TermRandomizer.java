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

package org.apache.mahout.classifier.sgd;

import org.apache.mahout.classifier.MurmurHash;
import org.apache.mahout.math.Vector;

import java.nio.charset.Charset;
import java.util.List;

/**
 * Interface for converting a list of terms into a vector in a deterministic
 * (pseudo) randomized way.
 */
public abstract class TermRandomizer {
  public abstract Vector randomizedInstance(List<String> terms, int window, boolean allPairs);

  /**
   * Hash a string and an integer into the range [0..numFeatures-1].
   *
   * @param term   The string.
   * @param probe  An integer that modifies the resulting hash.
   * @param numFeatures  The range into which the resulting hash must fit.
   * @return An integer in the range [0..numFeatures-1] that has good spread for small changes in term and probe.
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
   * @param term1   The first string.
   * @param term2   The second string.
   * @param probe  An integer that modifies the resulting hash.
   * @param numFeatures  The range into which the resulting hash must fit.
   * @return An integer in the range [0..numFeatures-1] that has good spread for small changes in term and probe.
   */
  protected int hash(String term1, String term2, int probe, int numFeatures) {
    long r = MurmurHash.hash64A(term1.getBytes(Charset.forName("UTF-8")), probe);
    r = MurmurHash.hash64A(term2.getBytes(Charset.forName("UTF-8")), (int) r) % numFeatures;
    if (r < 0) {
      r += numFeatures;
    }
    return (int) r;
  }
}
