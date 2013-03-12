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

package org.apache.mahout.classifier.df.split;

import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.DescriptorException;
import org.apache.mahout.classifier.df.data.conditions.Condition;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public final class RegressionSplitTest extends MahoutTestCase {

  private static Data[] generateTrainingData() throws DescriptorException {
    // Training data
    String[] trainData = new String[20];
    for (int i = 0; i < trainData.length; i++) {
      if (i % 3 == 0) {
        trainData[i] = "A," + (40 - i) + ',' + (i + 20);
      } else if (i % 3 == 1) {
        trainData[i] = "B," + (i + 20) + ',' + (40 - i);
      } else {
        trainData[i] = "C," + (i + 20) + ',' + (i + 20);
      }
    }
    // Dataset
    Dataset dataset = DataLoader.generateDataset("C N L", true, trainData);
    Data[] datas = new Data[3];
    datas[0] = DataLoader.loadData(dataset, trainData);

    // Training data
    trainData = new String[20];
    for (int i = 0; i < trainData.length; i++) {
      if (i % 2 == 0) {
        trainData[i] = "A," + (50 - i) + ',' + (i + 10);
      } else {
        trainData[i] = "B," + (i + 10) + ',' + (50 - i);
      }
    }
    datas[1] = DataLoader.loadData(dataset, trainData);

    // Training data
    trainData = new String[10];
    for (int i = 0; i < trainData.length; i++) {
      trainData[i] = "A," + (40 - i) + ',' + (i + 20);
    }
    datas[2] = DataLoader.loadData(dataset, trainData);

    return datas;
  }

  @Test
  public void testComputeSplit() throws DescriptorException {
    Data[] datas = generateTrainingData();

    RegressionSplit igSplit = new RegressionSplit();
    Split split = igSplit.computeSplit(datas[0], 1);
    assertEquals(180.0, split.getIg(), EPSILON);
    assertEquals(38.0, split.getSplit(), EPSILON);
    split = igSplit.computeSplit(datas[0].subset(Condition.lesser(1, 38.0)), 1);
    assertEquals(76.5, split.getIg(), EPSILON);
    assertEquals(21.5, split.getSplit(), EPSILON);

    split = igSplit.computeSplit(datas[1], 0);
    assertEquals(2205.0, split.getIg(), EPSILON);
    assertEquals(Double.NaN, split.getSplit(), EPSILON);
    split = igSplit.computeSplit(datas[1].subset(Condition.equals(0, 0.0)), 1);
    assertEquals(250.0, split.getIg(), EPSILON);
    assertEquals(41.0, split.getSplit(), EPSILON);
  }
}
