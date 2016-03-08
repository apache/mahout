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

package org.apache.mahout.classifier.df.tools;

import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.builder.DecisionTreeBuilder;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.node.CategoricalNode;
import org.apache.mahout.classifier.df.node.Leaf;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.classifier.df.node.NumericalNode;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.text.DecimalFormat;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;
@Deprecated
public final class VisualizerTest extends MahoutTestCase {
  
  private static final char DECIMAL_SEPARATOR =
      ((DecimalFormat) DecimalFormat.getInstance()).getDecimalFormatSymbols().getDecimalSeparator();
  
  private static final String[] TRAIN_DATA = {"sunny,85,85,FALSE,no",
      "sunny,80,90,TRUE,no", "overcast,83,86,FALSE,yes",
      "rainy,70,96,FALSE,yes", "rainy,68,80,FALSE,yes", "rainy,65,70,TRUE,no",
      "overcast,64,65,TRUE,yes", "sunny,72,95,FALSE,no",
      "sunny,69,70,FALSE,yes", "rainy,75,80,FALSE,yes", "sunny,75,70,TRUE,yes",
      "overcast,72,90,TRUE,yes", "overcast,81,75,FALSE,yes",
      "rainy,71,91,TRUE,no"};
  
  private static final String[] TEST_DATA = {"rainy,70,96,TRUE,-",
      "overcast,64,65,TRUE,-", "sunny,75,90,TRUE,-",};
  
  private static final String[] ATTRIBUTE_NAMES = {"outlook", "temperature",
      "humidity", "windy", "play"};
  
  private Random randomNumberGenerator;
  
  private Data trainingData;
  
  private Data testData;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    
    randomNumberGenerator = RandomUtils.getRandom(1);
    
    Dataset dataset = DataLoader
        .generateDataset("C N N C L", false, TRAIN_DATA);
    
    trainingData = DataLoader.loadData(dataset, TRAIN_DATA);
    
    testData = DataLoader.loadData(dataset, TEST_DATA);
  }
  
  @Test
  public void testTreeVisualize() throws Exception {
    // build tree
    DecisionTreeBuilder builder = new DecisionTreeBuilder();
    builder.setM(trainingData.getDataset().nbAttributes() - 1);
    Node tree = builder.build(randomNumberGenerator, trainingData);

    String visualization = TreeVisualizer.toString(tree, trainingData.getDataset(), ATTRIBUTE_NAMES);

    assertTrue(
        (String.format("\n" +
            "outlook = rainy\n" +
            "|   windy = FALSE : yes\n" +
            "|   windy = TRUE : no\n" +
            "outlook = sunny\n" +
            "|   humidity < 77%s5 : yes\n" +
            "|   humidity >= 77%s5 : no\n" +
                       "outlook = overcast : yes", DECIMAL_SEPARATOR, DECIMAL_SEPARATOR)).equals(visualization) ||
        (String.format("\n" +
            "outlook = rainy\n" +
            "|   windy = TRUE : no\n" +
            "|   windy = FALSE : yes\n" +
            "outlook = overcast : yes\n" +
            "outlook = sunny\n" +
            "|   humidity < 77%s5 : yes\n" +
            "|   humidity >= 77%s5 : no", DECIMAL_SEPARATOR, DECIMAL_SEPARATOR)).equals(visualization));
  }
  
  @Test
  public void testPredictTrace() throws Exception {
    // build tree
    DecisionTreeBuilder builder = new DecisionTreeBuilder();
    builder.setM(trainingData.getDataset().nbAttributes() - 1);
    Node tree = builder.build(randomNumberGenerator, trainingData);
    
    String[] prediction = TreeVisualizer.predictTrace(tree, testData,
			ATTRIBUTE_NAMES);
    Assert.assertArrayEquals(new String[] {
        "outlook = rainy -> windy = TRUE -> no", "outlook = overcast -> yes",
        String.format("outlook = sunny -> (humidity = 90) >= 77%s5 -> no", DECIMAL_SEPARATOR)}, prediction);
  }
  
  @Test
  public void testForestVisualize() throws Exception {
    // Tree
    NumericalNode root = new NumericalNode(2, 90, new Leaf(0),
        new CategoricalNode(0, new double[] {0, 1, 2}, new Node[] {
            new NumericalNode(1, 71, new Leaf(0), new Leaf(1)), new Leaf(1),
            new Leaf(0)}));
    List<Node> trees = new ArrayList<>();
    trees.add(root);

    // Forest
    DecisionForest forest = new DecisionForest(trees);
    String visualization = ForestVisualizer.toString(forest, trainingData.getDataset(), null);
    assertTrue(
        ("Tree[1]:\n2 < 90 : yes\n2 >= 90\n" +
            "|   0 = rainy\n" +
            "|   |   1 < 71 : yes\n" +
            "|   |   1 >= 71 : no\n" +
            "|   0 = sunny : no\n" +
            "|   0 = overcast : yes\n").equals(visualization) ||
        ("Tree[1]:\n" +
            "2 < 90 : no\n" +
            "2 >= 90\n" +
            "|   0 = rainy\n" +
            "|   |   1 < 71 : no\n" +
            "|   |   1 >= 71 : yes\n" +
            "|   0 = overcast : yes\n" +
            "|   0 = sunny : no\n").equals(visualization));

    visualization = ForestVisualizer.toString(forest, trainingData.getDataset(), ATTRIBUTE_NAMES);
    assertTrue(
        ("Tree[1]:\n" +
            "humidity < 90 : yes\n" +
            "humidity >= 90\n" +
            "|   outlook = rainy\n" +
            "|   |   temperature < 71 : yes\n" +
            "|   |   temperature >= 71 : no\n" +
            "|   outlook = sunny : no\n" +
            "|   outlook = overcast : yes\n").equals(visualization) ||
        ("Tree[1]:\n" +
            "humidity < 90 : no\n" +
            "humidity >= 90\n" +
            "|   outlook = rainy\n" +
            "|   |   temperature < 71 : no\n" +
            "|   |   temperature >= 71 : yes\n" +
            "|   outlook = overcast : yes\n" +
            "|   outlook = sunny : no\n").equals(visualization));
  }
  
  @Test
  public void testLeafless() throws Exception {
    List<Instance> instances = new ArrayList<>();
    for (int i = 0; i < trainingData.size(); i++) {
      if (trainingData.get(i).get(0) != 0.0d) {
        instances.add(trainingData.get(i));
      }
    }
    Data lessData = new Data(trainingData.getDataset(), instances);
    
    // build tree
    DecisionTreeBuilder builder = new DecisionTreeBuilder();
    builder.setM(trainingData.getDataset().nbAttributes() - 1);
    builder.setMinSplitNum(0);
    builder.setComplemented(false);
    Node tree = builder.build(randomNumberGenerator, lessData);

    String visualization = TreeVisualizer.toString(tree, trainingData.getDataset(), ATTRIBUTE_NAMES);
    assertTrue(
        (String.format("\noutlook = sunny\n" +
            "|   humidity < 77%s5 : yes\n" +
            "|   humidity >= 77%s5 : no\n" +
            "outlook = overcast : yes", DECIMAL_SEPARATOR, DECIMAL_SEPARATOR)).equals(visualization) ||
        (String.format("\noutlook = overcast : yes\n" +
            "outlook = sunny\n" +
            "|   humidity < 77%s5 : yes\n" +
            "|   humidity >= 77%s5 : no", DECIMAL_SEPARATOR, DECIMAL_SEPARATOR)).equals(visualization));
  }
  
  @Test
  public void testEmpty() throws Exception {
    Data emptyData = new Data(trainingData.getDataset());
    
    // build tree
    DecisionTreeBuilder builder = new DecisionTreeBuilder();
    Node tree = builder.build(randomNumberGenerator, emptyData);

    assertEquals(" : unknown", TreeVisualizer.toString(tree, trainingData.getDataset(), ATTRIBUTE_NAMES));
  }
}
