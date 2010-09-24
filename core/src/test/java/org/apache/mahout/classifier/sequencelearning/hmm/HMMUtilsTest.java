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

import junit.framework.Assert;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class HMMUtilsTest extends HMMTestBase {

  Matrix legal2_2;
  Matrix legal2_3;
  Matrix legal3_3;
  Vector legal2;
  Matrix illegal2_2;

  public void setUp() throws Exception {
    super.setUp();
    legal2_2 = new DenseMatrix(new double[][]{{0.5, 0.5}, {0.3, 0.7}});
    legal2_3 = new DenseMatrix(new double[][]{{0.2, 0.2, 0.6},
        {0.3, 0.3, 0.4}});
    legal3_3 = new DenseMatrix(new double[][]{{0.1, 0.1, 0.8},
        {0.1, 0.2, 0.7}, {0.2, 0.3, 0.5}});
    legal2 = new DenseVector(new double[]{0.4, 0.6});
    illegal2_2 = new DenseMatrix(new double[][]{{1, 2}, {3, 4}});
  }

  @Test
  public void testValidatorLegal() {
    HmmUtils.validate(new HmmModel(legal2_2, legal2_3, legal2));
  }

  @Test
  public void testValidatorDimensionError() {
    try {
      HmmUtils.validate(new HmmModel(legal3_3, legal2_3, legal2));
    } catch (IllegalArgumentException e) {
      // success
      return;
    }
    Assert.fail();
  }

  @Test
  public void testValidatorIllegelMatrixError() {
    try {
      HmmUtils.validate(new HmmModel(illegal2_2, legal2_3, legal2));
    } catch (IllegalArgumentException e) {
      // success
      return;
    }
    Assert.fail();
  }

  @Test
  public void testEncodeStateSequence() {
    String[] hiddenSequence = {"H1", "H2", "H0", "H3", "H4"};
    String[] outputSequence = {"O1", "O2", "O4", "O0"};
    // test encoding the hidden Sequence
    int[] hiddenSequenceEnc = HmmUtils.encodeStateSequence(model, Arrays
        .asList(hiddenSequence), false, -1);
    int[] outputSequenceEnc = HmmUtils.encodeStateSequence(model, Arrays
        .asList(outputSequence), true, -1);
    // expected state sequences
    int[] hiddenSequenceExp = {1, 2, 0, 3, -1};
    int[] outputSequenceExp = {1, 2, -1, 0};
    // compare
    for (int i = 0; i < hiddenSequenceEnc.length; ++i)
      Assert.assertEquals(hiddenSequenceExp[i], hiddenSequenceEnc[i]);
    for (int i = 0; i < outputSequenceEnc.length; ++i)
      Assert.assertEquals(outputSequenceExp[i], outputSequenceEnc[i]);
  }

  @Test
  public void testDecodeStateSequence() {
    int[] hiddenSequence = {1, 2, 0, 3, 10};
    int[] outputSequence = {1, 2, 10, 0};
    // test encoding the hidden Sequence
    java.util.Vector<String> hiddenSequenceDec = HmmUtils.decodeStateSequence(
        model, hiddenSequence, false, "unknown");
    java.util.Vector<String> outputSequenceDec = HmmUtils.decodeStateSequence(
        model, outputSequence, true, "unknown");
    // expected state sequences
    String[] hiddenSequenceExp = {"H1", "H2", "H0", "H3", "unknown"};
    String[] outputSequenceExp = {"O1", "O2", "unknown", "O0"};
    // compare
    for (int i = 0; i < hiddenSequenceExp.length; ++i)
      Assert.assertEquals(hiddenSequenceExp[i], hiddenSequenceDec.get(i));
    for (int i = 0; i < outputSequenceExp.length; ++i)
      Assert.assertEquals(outputSequenceExp[i], outputSequenceDec.get(i));
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
      if (i == 2)
        Assert.assertEquals(1.0, sparse_ip.getQuick(i));
      else
        Assert.assertEquals(0.0, sparse_ip.getQuick(i));
      for (int j = 0; j < sparseModel.getNrOfHiddenStates(); ++j) {
        if (i == j) {
          Assert.assertEquals(1.0, sparse_tr.getQuick(i, j));
          Assert.assertEquals(1.0, sparse_em.getQuick(i, j));
        } else {
          Assert.assertEquals(0.0, sparse_tr.getQuick(i, j));
          Assert.assertEquals(0.0, sparse_em.getQuick(i, j));
        }
      }
    }
  }

}
