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

package org.apache.mahout.df.mapreduce.partial;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Locale;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.DataLoader;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.data.Instance;
import org.apache.mahout.df.data.Utils;
import org.apache.mahout.df.mapreduce.Builder;
import org.apache.mahout.df.node.Node;
import org.junit.Test;

public final class PartitionBugTest extends MahoutTestCase {
  static final int NUM_ATTRIBUTES = 40;

  static final int NUM_INSTANCES = 200;

  static final int NUM_TREES = 10;

  static final int NUM_MAPS = 5;
    
  /**
   * Make sure that the correct instance ids are being computed
   */
  @Test
  public void testProcessOutput() throws Exception {
    Random rng = RandomUtils.getRandom();
    //long seed = rng.nextLong();

    // create a dataset large enough to be split up
    String descriptor = Utils.randomDescriptor(rng, NUM_ATTRIBUTES);
    double[][] source = Utils.randomDoubles(rng, descriptor, NUM_INSTANCES);

    // each instance label is its index in the dataset
    int labelId = Utils.findLabel(descriptor);
    for (int index = 0; index < NUM_INSTANCES; index++) {
      source[index][labelId] = index;
    }

    // store the data into a file
    String[] sData = Utils.double2String(source);
    Path dataPath = Utils.writeDataToTestFile(sData);
    Dataset dataset = DataLoader.generateDataset(descriptor, sData);
    Data data = DataLoader.loadData(dataset, sData);

    Configuration conf = new Configuration();
    Step0JobTest.setMaxSplitSize(conf, dataPath, NUM_MAPS);

    // prepare a custom TreeBuilder that will classify each
    // instance with its own label (in this case its index in the dataset)
    TreeBuilder treeBuilder = new MockTreeBuilder();
    
    // disable the second step because we can test without it
    // and we won't be able to serialize the MockNode
    PartialBuilder.setStep2(conf, false);
    long seed = 1L;
    Builder builder = new PartialSequentialBuilder(treeBuilder, dataPath, dataset, seed, conf);

    // remove the output path (its only used for testing)
    Path outputPath = builder.getOutputPath(conf);
    HadoopUtil.delete(conf, outputPath);

    builder.build(NUM_TREES, new MockCallback(data));
  }

  /**
   * Asserts that the instanceId are correct
   *
   */
  private static class MockCallback implements PredictionCallback {
    private final Data data;

    private MockCallback(Data data) {
      this.data = data;
    }

    @Override
    public void prediction(int treeId, int instanceId, int prediction) {
      // because of the bagging, prediction can be -1
      if (prediction == -1) {
        return;
      }

      assertEquals(String.format(Locale.ENGLISH, "treeId: %d, InstanceId: %d, Prediction: %d",
          treeId, instanceId, prediction), data.get(instanceId).getLabel(), prediction);
    }

  }

  /**
   * Custom Leaf node that returns for each instance its own label
   * 
   */
  private static class MockLeaf extends Node {

    @Override
    public int classify(Instance instance) {
      return instance.getLabel();
    }

    @Override
    protected String getString() {
      return "[MockLeaf]";
    }

    @Override
    public long maxDepth() {
      return 0;
    }

    @Override
    protected Type getType() {
      return Type.MOCKLEAF;
    }

    @Override
    public long nbNodes() {
      return 0;
    }

    @Override
    protected void writeNode(DataOutput out) throws IOException {
    }

    @Override
    public void readFields(DataInput in) throws IOException {
    }

    
  }

  private static class MockTreeBuilder implements TreeBuilder {

    @Override
    public Node build(Random rng, Data data) {
      return new MockLeaf();
    }

  }
}
