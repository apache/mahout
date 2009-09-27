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

package org.apache.mahout.df.split;

import static org.apache.mahout.df.data.Utils.double2String;
import static org.apache.mahout.df.data.Utils.randomDescriptor;
import static org.apache.mahout.df.data.Utils.randomDoublesWithSameLabel;

import java.util.Random;

import junit.framework.TestCase;

import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.DataLoader;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.data.Utils;
import org.apache.mahout.df.split.DefaultIgSplit;

public class DefaultIgSplitTest extends TestCase {

  protected final int nbAttributes = 10;

  public void testEntropy() throws Exception {
    Random rng = new Random();
    String descriptor = randomDescriptor(rng, nbAttributes);
    int label = Utils.findLabel(descriptor);
    double[][] temp;
    String[] sData;
    Data data;
    Dataset dataset;
    DefaultIgSplit iG;
    
    // all the vectors have the same label (0)
    temp = randomDoublesWithSameLabel(rng, descriptor, 100, 0);
    sData = double2String(temp);
    dataset = DataLoader.generateDataset(descriptor, sData);
    data = DataLoader.loadData(dataset, sData);
    iG = new DefaultIgSplit();
    
    double expected = 0.0 - 1.0 * Math.log(1.0) / Math.log(2.0);
    assertEquals(expected, iG.entropy(data));

    // 50/100 of the vectors have the label (1)
    // 50/100 of the vectors have the label (0)
    for (int index = 0; index < 50; index++) {
      temp[index][label] = 1.0;
    }
    sData = double2String(temp);
    dataset = DataLoader.generateDataset(descriptor, sData);
    data = DataLoader.loadData(dataset, sData);
    iG = new DefaultIgSplit();
    
    expected = 2.0 * -0.5 * Math.log(0.5) / Math.log(2.0);
    assertEquals(expected, iG.entropy(data));

    // 15/100 of the vectors have the label (2)
    // 35/100 of the vectors have the label (1)
    // 50/100 of the vectors have the label (0)
    for (int index = 0; index < 15; index++) {
      temp[index][label] = 2.0;
    }
    sData = double2String(temp);
    dataset = DataLoader.generateDataset(descriptor, sData);
    data = DataLoader.loadData(dataset, sData);
    iG = new DefaultIgSplit();
    
    expected = -0.15 * Math.log(0.15) / Math.log(2.0) - 0.35 * Math.log(0.35)
        / Math.log(2.0) - 0.5 * Math.log(0.5) / Math.log(2.0);
    assertEquals(expected, iG.entropy(data));
  }
}
