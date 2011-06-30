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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Implements a linear combination of L1 and L2 priors.  This can give an
 * interesting mixture of sparsity and load-sharing between redundant predictors.
 */
public class ElasticBandPrior implements PriorFunction {
  private double alphaByLambda;
  private L1 l1;
  private L2 l2;

  // Exists for Writable
  public ElasticBandPrior() {
    this(0.0);
  }

  public ElasticBandPrior(double alphaByLambda) {
    this.alphaByLambda = alphaByLambda;
    l1 = new L1();
    l2 = new L2(1);
  }

  @Override
  public double age(double oldValue, double generations, double learningRate) {
    oldValue *= Math.pow(1 - alphaByLambda * learningRate, generations);
    double newValue = oldValue - Math.signum(oldValue) * learningRate * generations;
    if (newValue * oldValue < 0.0) {
      // don't allow the value to change sign
      return 0.0;
    } else {
      return newValue;
    }
  }

  @Override
  public double logP(double betaIJ) {
    return l1.logP(betaIJ) + alphaByLambda * l2.logP(betaIJ);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(alphaByLambda);
    l1.write(out);
    l2.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    alphaByLambda = in.readDouble();
    l1 = new L1();
    l1.readFields(in);
    l2 = new L2();
    l2.readFields(in);
  }
}
