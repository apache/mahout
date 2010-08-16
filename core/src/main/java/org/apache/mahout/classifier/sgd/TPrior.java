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

import static java.lang.Math.log;
import static org.apache.commons.math.special.Gamma.logGamma;

/**
 * Provides a t-distribution as a prior.
 */
public class TPrior extends PriorFunction {
  private double df;

  public TPrior(double df) {
    this.df = df;
  }

  @Override
  public double age(double oldValue, double generations, double learningRate) {
    for (int i = 0; i < generations; i++) {
      oldValue = oldValue - learningRate * oldValue * (df + 1) / (df + oldValue * oldValue);
    }
    return oldValue;
  }

  @Override
  public double logP(double beta_ij) {
    return logGamma((df + 1) / 2) - log(df * Math.PI) - logGamma(df / 2) - (df + 1) / 2 * log(1 + beta_ij * beta_ij);
  }
}
