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
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

/**
 * Implements the basic logistic training law.
 */
public class DefaultGradient implements Gradient {
  /**
   * Provides a default gradient computation useful for logistic regression.  
   *
   * @param groupKey     A grouping key to allow per-something AUC loss to be used for training.
   * @param actual       The target variable value.
   * @param instance     The current feature vector to use for gradient computation
   * @param classifier   The classifier that can compute scores
   * @return  The gradient to be applied to beta
   */
  @Override
  public final Vector apply(String groupKey, int actual, Vector instance, AbstractVectorClassifier classifier) {
    // what does the current model say?
    Vector v = classifier.classify(instance);

    Vector r = v.like();
    if (actual != 0) {
      r.setQuick(actual - 1, 1);
    }
    r.assign(v, Functions.MINUS);
    return r;
  }
}
