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

import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Vector;

import java.util.Random;

/**
 * <p>Provides a stochastic mixture of ranking updates and normal logistic updates. This uses a
 * combination of AUC driven learning to improve ranking performance and traditional log-loss driven
 * learning to improve log-likelihood.</p>
 *
 * <p>See www.eecs.tufts.edu/~dsculley/papers/combined-ranking-and-regression.pdf</p>
 *
 * <p>This implementation only makes sense for the binomial case.</p>
 */
public class MixedGradient implements Gradient {

  private final double alpha;
  private final RankingGradient rank;
  private final Gradient basic;
  private final Random random = RandomUtils.getRandom();
  private boolean hasZero;
  private boolean hasOne;

  public MixedGradient(double alpha, int window) {
    this.alpha = alpha;
    this.rank = new RankingGradient(window);
    this.basic = this.rank.getBaseGradient();
  }

  @Override
  public Vector apply(String groupKey, int actual, Vector instance, AbstractVectorClassifier classifier) {
    if (random.nextDouble() < alpha) {
      // one option is to apply a ranking update relative to our recent history
      if (!hasZero || !hasOne) {
        throw new IllegalStateException();
      }
      return rank.apply(groupKey, actual, instance, classifier);
    } else {
      hasZero |= actual == 0;
      hasOne |= actual == 1;
      // the other option is a normal update, but we have to update our history on the way
      rank.addToHistory(actual, instance);
      return basic.apply(groupKey, actual, instance, classifier);
    }
  }
}
