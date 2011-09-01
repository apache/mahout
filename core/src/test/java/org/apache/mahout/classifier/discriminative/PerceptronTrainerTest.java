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

package org.apache.mahout.classifier.discriminative;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Before;
import org.junit.Test;

public final class PerceptronTrainerTest extends MahoutTestCase {

  private PerceptronTrainer trainer;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    trainer = new PerceptronTrainer(3, 0.5, 0.1, 1.0, 1.0);
  }

  @Test
  public void testUpdate() throws Exception {
    double[] labels = { 1.0, 1.0, 1.0, 0.0 };
    Vector labelset = new DenseVector(labels);
    double[][] values = new double[3][4];
    for (int i = 0; i < 3; i++) {
      values[i][0] = 1.0;
      values[i][1] = 1.0;
      values[i][2] = 1.0;
      values[i][3] = 1.0;
    }
    values[1][0] = 0.0;
    values[2][0] = 0.0;
    values[1][1] = 0.0;
    values[2][2] = 0.0;

    Matrix dataset = new DenseMatrix(values);
    this.trainer.train(labelset, dataset);
    assertFalse(this.trainer.getModel().classify(dataset.viewColumn(3)));
    assertTrue(this.trainer.getModel().classify(dataset.viewColumn(0)));
  }

}
