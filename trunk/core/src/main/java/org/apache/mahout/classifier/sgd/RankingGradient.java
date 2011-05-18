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

import com.google.common.collect.Lists;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

/**
 * Uses the difference between this instance and recent history to get a
 * gradient that optimizes ranking performance.  Essentially this is the
 * same as directly optimizing AUC.  It isn't expected that this would
 * be used alone, but rather that a MixedGradient would use it and a
 * DefaultGradient together to combine both ranking and log-likelihood
 * goals.
 */
public class RankingGradient implements Gradient {

  private static final Gradient BASIC = new DefaultGradient();

  private int window = 10;

  private final List<Deque<Vector>> history = Lists.newArrayList();

  public RankingGradient(int window) {
    this.window = window;
  }

  @Override
  public final Vector apply(String groupKey, int actual, Vector instance, AbstractVectorClassifier classifier) {
    addToHistory(actual, instance);

    // now compute average gradient versus saved vectors from the other side
    Deque<Vector> otherSide = history.get(1 - actual);
    int n = otherSide.size();

    Vector r = null;
    for (Vector other : otherSide) {
      Vector g = BASIC.apply(groupKey, actual, instance.minus(other), classifier);

      if (r == null) {
        r = g;
      } else {
        r.assign(g, Functions.plusMult(1.0 / n));
      }
    }
    return r;
  }

  public void addToHistory(int actual, Vector instance) {
    while (history.size() <= actual) {
      history.add(new ArrayDeque<Vector>(window));
    }
    // save this instance
    Deque<Vector> ourSide = history.get(actual);
    ourSide.add(instance);
    while (ourSide.size() >= window) {
      ourSide.pollFirst();
    }
  }

  public Gradient getBaseGradient() {
    return BASIC;
  }
}
