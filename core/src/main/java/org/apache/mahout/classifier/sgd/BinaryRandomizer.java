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

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * Multiplies a sparse vector in the form of a term list by a random binary matrix.
 */
public class BinaryRandomizer extends TermRandomizer {
  private int probes;
  private int numFeatures;

  public BinaryRandomizer(int probes, int numFeatures) {
    this.probes = probes;
    this.numFeatures = numFeatures;
  }

  @Override
  public Vector randomizedInstance(List<String> terms, int window, boolean allPairs) {
    Vector instance = new RandomAccessSparseVector(getNumFeatures(), Math.min(terms.size() * getProbes(), 20));
    int n = 0;
    for (String term : terms) {
      for (int probe = 0; probe < getProbes(); probe++) {
        int i = hash(term, probe, getNumFeatures());
        instance.setQuick(i, instance.get(i) + 1);
      }

      if (allPairs) {
        for (String other : terms) {
          for (int probe = 0; probe < getProbes(); probe++) {
            int i = hash(term, other, probe, getNumFeatures());
            instance.setQuick(i, instance.get(i) + 1);
          }
        }
      }

      if (window > 0) {
        for (int j = Math.max(0, n - window); j < n; j++) {
          for (int probe = 0; probe < getProbes(); probe++) {
            int i = hash(term, terms.get(j), probe, getNumFeatures());
            instance.setQuick(i, instance.get(i) + 1);
          }
        }
      }
      n++;
    }
    return instance;
  }

  public int getProbes() {
    return probes;
  }

  public int getNumFeatures() {
    return numFeatures;
  }
}
