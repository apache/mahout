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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

public final class ConfusionMatrixTest extends MahoutTestCase {

  private static final int[][] VALUES = {{2, 3}, {10, 20}};
  private static final String[] LABELS = {"Label1", "Label2"};
  private static final String DEFAULT_LABEL = "other";
  
  @Test
  public void testBuild() {
    ConfusionMatrix cm = fillCM(VALUES, LABELS, DEFAULT_LABEL);
    checkValues(cm);
    checkAccuracy(cm);
  }

  @Test
  public void testGetMatrix() {
	    ConfusionMatrix cm = fillCM(VALUES, LABELS, DEFAULT_LABEL);
	    Matrix m = cm.getMatrix();
	    Map<String, Integer> rowLabels = m.getRowLabelBindings();
	    assertEquals(cm.getLabels().size(), m.numCols());
	    assertTrue(rowLabels.keySet().contains(LABELS[0]));
	    assertTrue(rowLabels.keySet().contains(LABELS[1]));
	    assertTrue(rowLabels.keySet().contains(DEFAULT_LABEL));
	    assertEquals(2, cm.getCorrect(LABELS[0]));
	    assertEquals(20, cm.getCorrect(LABELS[1]));
	    assertEquals(0, cm.getCorrect(DEFAULT_LABEL));
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
    assertEquals(0, counts[0][2]);
    assertEquals(0, counts[1][2]);
    assertEquals(3, cm.getLabels().size());
    assertTrue(cm.getLabels().contains(LABELS[0]));
    assertTrue(cm.getLabels().contains(LABELS[1]));
    assertTrue(cm.getLabels().contains(DEFAULT_LABEL));

  }

  private static void checkAccuracy(ConfusionMatrix cm) {
    Collection<String> labelstrs = cm.getLabels();
    assertEquals(3, labelstrs.size());
    assertEquals(40.0, cm.getAccuracy("Label1"), EPSILON);
    assertEquals(66.666666667, cm.getAccuracy("Label2"), EPSILON);
    assertTrue(Double.isNaN(cm.getAccuracy("other")));
  }
  
  private static ConfusionMatrix fillCM(int[][] values, String[] labels, String defaultLabel) {
    Collection<String> labelList = new ArrayList<String>();
    labelList.add(labels[0]);
    labelList.add(labels[1]);
    ConfusionMatrix cm = new ConfusionMatrix(labelList, defaultLabel);
    int[][] v = cm.getConfusionMatrix();
    v[0][0] = values[0][0];
    v[0][1] = values[0][1];
    v[1][0] = values[1][0];
    v[1][1] = values[1][1];
    return cm;
  }
  
}
