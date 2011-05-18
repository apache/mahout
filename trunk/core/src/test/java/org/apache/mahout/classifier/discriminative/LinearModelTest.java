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
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Before;
import org.junit.Test;

public final class LinearModelTest extends MahoutTestCase {

  private LinearModel model;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    double[] values = {0.0, 1.0, 0.0, 1.0, 0.0};
    Vector hyperplane = new DenseVector(values);
    this.model = new LinearModel(hyperplane, 0.1, 0.5);
  }

  @Test
  public void testClassify() {
    double[] valuesFalse = {1.0, 0.0, 1.0, 0.0, 1.0};
    Vector dataPointFalse = new DenseVector(valuesFalse);
    assertFalse(this.model.classify(dataPointFalse));

    double[] valuesTrue = {0.0, 1.0, 0.0, 1.0, 0.0};
    Vector dataPointTrue = new DenseVector(valuesTrue);
    assertTrue(this.model.classify(dataPointTrue));
  }

  @Test
  public void testAddDelta() {
    double[] values = {1.0, -1.0, 1.0, -1.0, 1.0};
    this.model.addDelta(new DenseVector(values));

    double[] valuesFalse = {1.0, 0.0, 1.0, 0.0, 1.0};
    Vector dataPointFalse = new DenseVector(valuesFalse);
    assertTrue(this.model.classify(dataPointFalse));

    double[] valuesTrue = {0.0, 1.0, 0.0, 1.0, 0.0};
    Vector dataPointTrue = new DenseVector(valuesTrue);
    assertFalse(this.model.classify(dataPointTrue));
  }

  @Test
  public void testTimesDelta() {
    double[] values = {-1.0, -1.0, -1.0, -1.0, -1.0};
    this.model.addDelta(new DenseVector(values));
    double[] dotval = {-1.0, -1.0, -1.0, -1.0, -1.0};
    
    for (int i = 0; i < dotval.length; i++) {
      this.model.timesDelta(i, dotval[i]);
    }

    double[] valuesFalse = {1.0, 0.0, 1.0, 0.0, 1.0};
    Vector dataPointFalse = new DenseVector(valuesFalse);
    assertTrue(this.model.classify(dataPointFalse));

    double[] valuesTrue = {0.0, 1.0, 0.0, 1.0, 0.0};
    Vector dataPointTrue = new DenseVector(valuesTrue);
    assertFalse(this.model.classify(dataPointTrue));
  }

}
