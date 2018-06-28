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

import java.util.Arrays;
import java.util.List;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class HMMUtilsTest extends HMMTestBase {

  private Matrix legal22;
  private Matrix legal23;
  private Matrix legal33;
  private Vector legal2;
  private Matrix illegal22;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    legal22 = new DenseMatrix(new double[][]{{0.5, 0.5}, {0.3, 0.7}});
    legal23 = new DenseMatrix(new double[][]{{0.2, 0.2, 0.6},
        {0.3, 0.3, 0.4}});
    legal33 = new DenseMatrix(new double[][]{{0.1, 0.1, 0.8},
        {0.1, 0.2, 0.7}, {0.2, 0.3, 0.5}});
    legal2 = new DenseVector(new double[]{0.4, 0.6});
    illegal22 = new DenseMatrix(new double[][]{{1, 2}, {3, 4}});
  }

  @Test
  public void testValidatorLegal() {
    HmmUtils.validate(new HmmModel(legal22, legal23, legal2));
  }

  @Test
  public void testValidatorDimensionError() {
    try {
      HmmUtils.validate(new HmmModel(legal33, legal23, legal2));
    } catch (IllegalArgumentException e) {
      // success
      return;
    }
    fail();
  }

  @Test
  public void testValidatorIllegelMatrixError() {
    try {
      HmmUtils.validate(new HmmModel(illegal22, legal23, legal2));
    } catch (IllegalArgumentException e) {
      // success
      return;
    }
    fail();
  }

  @Test
  public void testEncodeStateSequence() {
    String[] hiddenSequence = {"H1", "H2", "H0", "H3", "H4"};
    String[] outputSequence = {"O1", "O2", "O4", "O0"};
    // test encoding the hidden Sequence
    int[] hiddenSequenceEnc = HmmUtils.encodeStateSequence(getModel(), Arrays
        .asList(hiddenSequence), false, -1);
    int[] outputSequenceEnc = HmmUtils.encodeStateSequence(getModel(), Arrays
        .asList(outputSequence), true, -1);
    // expected state sequences
    int[] hiddenSequenceExp = {1, 2, 0, 3, -1};
    int[] outputSequenceExp = {1, 2, -1, 0};
    // compare
    for (int i = 0; i < hiddenSequenceEnc.length; ++i) {
      assertEquals(hiddenSequenceExp[i], hiddenSequenceEnc[i]);
    }
    for (int i = 0; i < outputSequenceEnc.length; ++i) {
      assertEquals(outputSequenceExp[i], outputSequenceEnc[i]);
    }
  }

  @Test
  public void testDecodeStateSequence() {
    int[] hiddenSequence = {1, 2, 0, 3, 10};
    int[] outputSequence = {1, 2, 10, 0};
    // test encoding the hidden Sequence
    List<String> hiddenSequenceDec = HmmUtils.decodeStateSequence(
        getModel(), hiddenSequence, false, "unknown");
    List<String> outputSequenceDec = HmmUtils.decodeStateSequence(
        getModel(), outputSequence, true, "unknown");
    // expected state sequences
    String[] hiddenSequenceExp = {"H1", "H2", "H0", "H3", "unknown"};
    String[] outputSequenceExp = {"O1", "O2", "unknown", "O0"};
    // compare
    for (int i = 0; i < hiddenSequenceExp.length; ++i) {
      assertEquals(hiddenSequenceExp[i], hiddenSequenceDec.get(i));
    }
    for (int i = 0; i < outputSequenceExp.length; ++i) {
      assertEquals(outputSequenceExp[i], outputSequenceDec.get(i));
    }
  }

  @Test
  public void testNormalizeModel() {
    DenseVector ip = new DenseVector(new double[]{10, 20});
    DenseMatrix tr = new DenseMatrix(new double[][]{{10, 10}, {20, 25}});
    DenseMatrix em = new DenseMatrix(new double[][]{{5, 7}, {10, 15}});
    HmmModel model = new HmmModel(tr, em, ip);
    HmmUtils.normalizeModel(model);
    // the model should be valid now
    HmmUtils.validate(model);
  }

  @Test
  public void testTruncateModel() {
    DenseVector ip = new DenseVector(new double[]{0.0001, 0.0001, 0.9998});
    DenseMatrix tr = new DenseMatrix(new double[][]{
        {0.9998, 0.0001, 0.0001}, {0.0001, 0.9998, 0.0001},
        {0.0001, 0.0001, 0.9998}});
    DenseMatrix em = new DenseMatrix(new double[][]{
        {0.9998, 0.0001, 0.0001}, {0.0001, 0.9998, 0.0001},
        {0.0001, 0.0001, 0.9998}});
    HmmModel model = new HmmModel(tr, em, ip);
    // now truncate the model
    HmmModel sparseModel = HmmUtils.truncateModel(model, 0.01);
    // first make sure this is a valid model
    HmmUtils.validate(sparseModel);
    // now check whether the values are as expected
    Vector sparse_ip = sparseModel.getInitialProbabilities();
    Matrix sparse_tr = sparseModel.getTransitionMatrix();
    Matrix sparse_em = sparseModel.getEmissionMatrix();
    for (int i = 0; i < sparseModel.getNrOfHiddenStates(); ++i) {
      assertEquals(i == 2 ? 1.0 : 0.0, sparse_ip.getQuick(i), EPSILON);
      for (int j = 0; j < sparseModel.getNrOfHiddenStates(); ++j) {
        if (i == j) {
          assertEquals(1.0, sparse_tr.getQuick(i, j), EPSILON);
          assertEquals(1.0, sparse_em.getQuick(i, j), EPSILON);
        } else {
          assertEquals(0.0, sparse_tr.getQuick(i, j), EPSILON);
          assertEquals(0.0, sparse_em.getQuick(i, j), EPSILON);
        }
      }
    }
  }

}
