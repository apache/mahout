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

import java.util.Random;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Utils;
import org.junit.Test;

public final class DefaultIgSplitTest extends MahoutTestCase {

  private static final int NUM_ATTRIBUTES = 10;

  @Test
  public void testEntropy() throws Exception {
    Random rng = RandomUtils.getRandom();
    String descriptor = Utils.randomDescriptor(rng, NUM_ATTRIBUTES);
    int label = Utils.findLabel(descriptor);

    // all the vectors have the same label (0)
    double[][] temp = Utils.randomDoublesWithSameLabel(rng, descriptor, false, 100, 0);
    String[] sData = Utils.double2String(temp);
    Dataset dataset = DataLoader.generateDataset(descriptor, false, sData);
    Data data = DataLoader.loadData(dataset, sData);
    DefaultIgSplit iG = new DefaultIgSplit();

    double expected = 0.0 - 1.0 * Math.log(1.0) / Math.log(2.0);
    assertEquals(expected, iG.entropy(data), EPSILON);

    // 50/100 of the vectors have the label (1)
    // 50/100 of the vectors have the label (0)
    for (int index = 0; index < 50; index++) {
      temp[index][label] = 1.0;
    }
    sData = Utils.double2String(temp);
    dataset = DataLoader.generateDataset(descriptor, false, sData);
    data = DataLoader.loadData(dataset, sData);
    iG = new DefaultIgSplit();
    
    expected = 2.0 * -0.5 * Math.log(0.5) / Math.log(2.0);
    assertEquals(expected, iG.entropy(data), EPSILON);

    // 15/100 of the vectors have the label (2)
    // 35/100 of the vectors have the label (1)
    // 50/100 of the vectors have the label (0)
    for (int index = 0; index < 15; index++) {
      temp[index][label] = 2.0;
    }
    sData = Utils.double2String(temp);
    dataset = DataLoader.generateDataset(descriptor, false, sData);
    data = DataLoader.loadData(dataset, sData);
    iG = new DefaultIgSplit();
    
    expected = -0.15 * Math.log(0.15) / Math.log(2.0) - 0.35 * Math.log(0.35)
        / Math.log(2.0) - 0.5 * Math.log(0.5) / Math.log(2.0);
    assertEquals(expected, iG.entropy(data), EPSILON);
  }
}
