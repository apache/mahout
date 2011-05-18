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

import org.apache.hadoop.io.Writable;

/**
 * A prior is used to regularize the learning algorithm.  This allows a trade-off to
 * be made between complexity of the model being learned and the accuracy with which
 * the model fits the training data.  There are different definitions of complexity
 * which can be approximated using different priors.  For large sparse systems, such
 * as text classification, the L1 prior is often used which favors sparse models.
 */
public interface PriorFunction extends Writable {
  /**
   * Applies the regularization to a coefficient.
   * @param oldValue        The previous value.
   * @param generations     The number of generations.
   * @param learningRate    The learning rate with lambda baked in.
   * @return                The new coefficient value after regularization.
   */
  double age(double oldValue, double generations, double learningRate);

  /**
   * Returns the log of the probability of a particular coefficient value according to the prior.
   * @param betaIJ          The coefficient.
   * @return                The log probability.
   */
  double logP(double betaIJ);
}
