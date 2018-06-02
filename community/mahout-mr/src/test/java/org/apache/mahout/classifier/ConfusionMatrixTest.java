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

package org.apache.mahout.classifier;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

import com.google.common.collect.Lists;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

public final class ConfusionMatrixTest extends MahoutTestCase {

  private static final int[][] VALUES = {{2, 3}, {10, 20}};
  private static final String[] LABELS = {"Label1", "Label2"};
  private static final int[] OTHER = {3, 6};
  private static final String DEFAULT_LABEL = "other";
  
  @Test
  public void testBuild() {
    ConfusionMatrix confusionMatrix = fillConfusionMatrix(VALUES, LABELS, DEFAULT_LABEL);
    checkValues(confusionMatrix);
    checkAccuracy(confusionMatrix);
  }

  @Test
  public void testGetMatrix() {
    ConfusionMatrix confusionMatrix = fillConfusionMatrix(VALUES, LABELS, DEFAULT_LABEL);
    Matrix m = confusionMatrix.getMatrix();
    Map<String, Integer> rowLabels = m.getRowLabelBindings();

    assertEquals(confusionMatrix.getLabels().size(), m.numCols());
    assertTrue(rowLabels.keySet().contains(LABELS[0]));
    assertTrue(rowLabels.keySet().contains(LABELS[1]));
    assertTrue(rowLabels.keySet().contains(DEFAULT_LABEL));
    assertEquals(2, confusionMatrix.getCorrect(LABELS[0]));
    assertEquals(20, confusionMatrix.getCorrect(LABELS[1]));
    assertEquals(0, confusionMatrix.getCorrect(DEFAULT_LABEL));
  }

    /**
     * Example taken from
     * http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
     */
    @Test
    public void testPrecisionRecallAndF1ScoreAsScikitLearn() {
      Collection<String> labelList = Arrays.asList("0", "1", "2");

      ConfusionMatrix confusionMatrix = new ConfusionMatrix(labelList, "DEFAULT");
      confusionMatrix.putCount("0", "0", 2);
      confusionMatrix.putCount("1", "0", 1);
      confusionMatrix.putCount("1", "2", 1);
      confusionMatrix.putCount("2", "1", 2);

      double delta = 0.001;
      assertEquals(0.222, confusionMatrix.getWeightedPrecision(), delta);
      assertEquals(0.333, confusionMatrix.getWeightedRecall(), delta);
      assertEquals(0.266, confusionMatrix.getWeightedF1score(), delta);
    }

  private static void checkValues(ConfusionMatrix cm) {
    int[][] counts = cm.getConfusionMatrix();
    cm.toString();
    assertEquals(counts.length, counts[0].length);
    assertEquals(3, counts.length);
    assertEquals(VALUES[0][0], counts[0][0]);
    assertEquals(VALUES[0][1], counts[0][1]);
    assertEquals(VALUES[1][0], counts[1][0]);
    assertEquals(VALUES[1][1], counts[1][1]);
    assertTrue(Arrays.equals(new int[3], counts[2])); // zeros
    assertEquals(OTHER[0], counts[0][2]);
    assertEquals(OTHER[1], counts[1][2]);
    assertEquals(3, cm.getLabels().size());
    assertTrue(cm.getLabels().contains(LABELS[0]));
    assertTrue(cm.getLabels().contains(LABELS[1]));
    assertTrue(cm.getLabels().contains(DEFAULT_LABEL));
  }

  private static void checkAccuracy(ConfusionMatrix cm) {
    Collection<String> labelstrs = cm.getLabels();
    assertEquals(3, labelstrs.size());
    assertEquals(25.0, cm.getAccuracy("Label1"), EPSILON);
    assertEquals(55.5555555, cm.getAccuracy("Label2"), EPSILON);
    assertTrue(Double.isNaN(cm.getAccuracy("other")));
  }
  
  private static ConfusionMatrix fillConfusionMatrix(int[][] values, String[] labels, String defaultLabel) {
    Collection<String> labelList = Lists.newArrayList();
    labelList.add(labels[0]);
    labelList.add(labels[1]);
    ConfusionMatrix confusionMatrix = new ConfusionMatrix(labelList, defaultLabel);

    confusionMatrix.putCount("Label1", "Label1", values[0][0]);
    confusionMatrix.putCount("Label1", "Label2", values[0][1]);
    confusionMatrix.putCount("Label2", "Label1", values[1][0]);
    confusionMatrix.putCount("Label2", "Label2", values[1][1]);
    confusionMatrix.putCount("Label1", DEFAULT_LABEL, OTHER[0]);
    confusionMatrix.putCount("Label2", DEFAULT_LABEL, OTHER[1]);
    return confusionMatrix;
  }

}
