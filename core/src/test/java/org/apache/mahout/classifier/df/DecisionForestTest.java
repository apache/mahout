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

package org.apache.mahout.classifier.df;

import java.util.List;
import java.util.Random;

import org.apache.mahout.classifier.df.builder.DecisionTreeBuilder;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.DescriptorException;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import com.google.common.collect.Lists;

public final class DecisionForestTest extends MahoutTestCase {

  private static final String[] TRAIN_DATA = {"sunny,85,85,FALSE,no",
    "sunny,80,90,TRUE,no", "overcast,83,86,FALSE,yes",
    "rainy,70,96,FALSE,yes", "rainy,68,80,FALSE,yes", "rainy,65,70,TRUE,no",
    "overcast,64,65,TRUE,yes", "sunny,72,95,FALSE,no",
    "sunny,69,70,FALSE,yes", "rainy,75,80,FALSE,yes", "sunny,75,70,TRUE,yes",
    "overcast,72,90,TRUE,yes", "overcast,81,75,FALSE,yes",
    "rainy,71,91,TRUE,no"};
  
  private static final String[] TEST_DATA = {"rainy,70,96,TRUE,-",
    "overcast,64,65,TRUE,-", "sunny,75,90,TRUE,-",};

  private Random rng;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    rng = RandomUtils.getRandom();
  }

  private static Data[] generateTrainingDataA() throws DescriptorException {
    // Dataset
    Dataset dataset = DataLoader.generateDataset("C N N C L", false, TRAIN_DATA);
    
    // Training data
    Data data = DataLoader.loadData(dataset, TRAIN_DATA);
    @SuppressWarnings("unchecked")
    List<Instance>[] instances = new List[3];
    for (int i = 0; i < instances.length; i++) {
      instances[i] = Lists.newArrayList();
    }
    for (int i = 0; i < data.size(); i++) {
      if (data.get(i).get(0) == 0.0d) {
        instances[0].add(data.get(i));
      } else {
        instances[1].add(data.get(i));
      }
    }
    Data[] datas = new Data[instances.length];
    for (int i = 0; i < datas.length; i++) {
      datas[i] = new Data(dataset, instances[i]);
    }

    return datas;
  }

  private static Data[] generateTrainingDataB() throws DescriptorException {

    // Training data
    String[] trainData = new String[20];
    for (int i = 0; i < trainData.length; i++) {
      if (i % 3 == 0) {
        trainData[i] = "A," + (40 - i) + ',' +  (i + 20);
      } else if (i % 3 == 1) {
        trainData[i] = "B," + (i + 20) + ',' +  (40 - i);
      } else {
        trainData[i] = "C," + (i + 20) + ',' +  (i + 20);
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
        trainData[i] = "A," + (50 - i) + ',' +  (i + 10);
      } else {
        trainData[i] = "B," + (i + 10) + ',' +  (50 - i);
      }
    }
    datas[1] = DataLoader.loadData(dataset, trainData);

    // Training data
    trainData = new String[10];
    for (int i = 0; i < trainData.length; i++) {
      trainData[i] = "A," + (40 - i) + ',' +  (i + 20);
    }
    datas[2] = DataLoader.loadData(dataset, trainData);

    return datas;
  }
  
  private DecisionForest buildForest(Data[] datas) {
    List<Node> trees = Lists.newArrayList();
    for (Data data : datas) {
      // build tree
      DecisionTreeBuilder builder = new DecisionTreeBuilder();
      builder.setM(data.getDataset().nbAttributes() - 1);
      builder.setMinSplitNum(0);
      builder.setComplemented(false);
      trees.add(builder.build(rng, data));
    }
    return new DecisionForest(trees);
  }
  
  @Test
  public void testClassify() throws DescriptorException {
    // Training data
    Data[] datas = generateTrainingDataA();
    // Build Forest
    DecisionForest forest = buildForest(datas);
    // Test data
    Data testData = DataLoader.loadData(datas[0].getDataset(), TEST_DATA);

    assertEquals(1.0, forest.classify(testData.getDataset(), rng, testData.get(0)), EPSILON);
    // This one is tie-broken -- 1 is OK too
    assertEquals(0.0, forest.classify(testData.getDataset(), rng, testData.get(1)), EPSILON);
    assertEquals(1.0, forest.classify(testData.getDataset(), rng, testData.get(2)), EPSILON);
  }

  @Test
  public void testClassifyData() throws DescriptorException {
    // Training data
    Data[] datas = generateTrainingDataA();
    // Build Forest
    DecisionForest forest = buildForest(datas);
    // Test data
    Data testData = DataLoader.loadData(datas[0].getDataset(), TEST_DATA);

    double[][] predictions = new double[testData.size()][];
    forest.classify(testData, predictions);
    assertArrayEquals(new double[][]{{1.0, Double.NaN, Double.NaN},
        {1.0, 0.0, Double.NaN}, {1.0, 1.0, Double.NaN}}, predictions);
  }

  @Test
  public void testRegression() throws DescriptorException {
    Data[] datas = generateTrainingDataB();
    DecisionForest[] forests = new DecisionForest[datas.length];
    for (int i = 0; i < datas.length; i++) {
      Data[] subDatas = new Data[datas.length - 1];
      int k = 0;
      for (int j = 0; j < datas.length; j++) {
        if (j != i) {
          subDatas[k] = datas[j];
          k++;
        }
      }
      forests[i] = buildForest(subDatas);
    }
    
    double[][] predictions = new double[datas[0].size()][];
    forests[0].classify(datas[0], predictions);
    assertArrayEquals(new double[]{20.0, 20.0}, predictions[0], EPSILON);
    assertArrayEquals(new double[]{39.0, 29.0}, predictions[1], EPSILON);
    assertArrayEquals(new double[]{Double.NaN, 29.0}, predictions[2], EPSILON);
    assertArrayEquals(new double[]{Double.NaN, 23.0}, predictions[17], EPSILON);

    predictions = new double[datas[1].size()][];
    forests[1].classify(datas[1], predictions);
    assertArrayEquals(new double[]{30.0, 29.0}, predictions[19], EPSILON);

    predictions = new double[datas[2].size()][];
    forests[2].classify(datas[2], predictions);
    assertArrayEquals(new double[]{29.0, 28.0}, predictions[9], EPSILON);

    assertEquals(20.0, forests[0].classify(datas[0].getDataset(), rng, datas[0].get(0)), EPSILON);
    assertEquals(34.0, forests[0].classify(datas[0].getDataset(), rng, datas[0].get(1)), EPSILON);
    assertEquals(29.0, forests[0].classify(datas[0].getDataset(), rng, datas[0].get(2)), EPSILON);
  }
}
