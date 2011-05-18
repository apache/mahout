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
 * A uniform prior.  This is an improper prior that corresponds to no regularization at all.
 */
public class UniformPrior implements PriorFunction {
  @Override
  public double age(double oldValue, double generations, double learningRate) {
    return oldValue;
  }

  @Override
  public double logP(double betaIJ) {
    return 0;
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    // nothing to write
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    // stateless class is trivial to read
  }
}
