/**
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

package org.apache.mahout.classifier.sequencelearning.hmm;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;

public class HMMTestBase extends MahoutTestCase {

  private HmmModel model;
  private final int[] sequence = {1, 0, 2, 2, 0, 0, 1};

  /**
   * We initialize a new HMM model using the following parameters # hidden
   * states: 4 ("H0","H1","H2","H3") # output states: 3 ("O0","O1","O2") #
   * transition matrix to: H0 H1 H2 H3 from: H0 0.5 0.1 0.1 0.3 H1 0.4 0.4 0.1
   * 0.1 H2 0.1 0.0 0.8 0.1 H3 0.1 0.1 0.1 0.7 # output matrix to: O0 O1 O2
   * from: H0 0.8 0.1 0.1 H1 0.6 0.1 0.3 H2 0.1 0.8 0.1 H3 0.0 0.1 0.9 # initial
   * probabilities H0 0.2
   * <p/>
   * H1 0.1 H2 0.4 H3 0.3
   * <p/>
   * We also intialize an observation sequence: "O1" "O0" "O2" "O2" "O0" "O0"
   * "O1"
   */

  @Override
  public void setUp() throws Exception {
    super.setUp();
    // intialize the hidden/output state names
    String[] hiddenNames = {"H0", "H1", "H2", "H3"};
    String[] outputNames = {"O0", "O1", "O2"};
    // initialize the transition matrix
    double[][] transitionP = {{0.5, 0.1, 0.1, 0.3}, {0.4, 0.4, 0.1, 0.1},
        {0.1, 0.0, 0.8, 0.1}, {0.1, 0.1, 0.1, 0.7}};
    // initialize the emission matrix
    double[][] emissionP = {{0.8, 0.1, 0.1}, {0.6, 0.1, 0.3},
        {0.1, 0.8, 0.1}, {0.0, 0.1, 0.9}};
    // initialize the initial probability vector
    double[] initialP = {0.2, 0.1, 0.4, 0.3};
    // now generate the model
    model = new HmmModel(new DenseMatrix(transitionP), new DenseMatrix(
        emissionP), new DenseVector(initialP));
    model.registerHiddenStateNames(hiddenNames);
    model.registerOutputStateNames(outputNames);
    // make sure the model is valid :)
    HmmUtils.validate(model);
  }

  protected HmmModel getModel() {
    return model;
  }

  protected int[] getSequence() {
    return sequence;
  }
}
