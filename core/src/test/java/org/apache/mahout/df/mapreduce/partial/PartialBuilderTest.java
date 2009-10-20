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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import junit.framework.TestCase;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapreduce.Job;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.builder.DefaultTreeBuilder;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.node.Leaf;
import org.apache.mahout.df.node.Node;

public class PartialBuilderTest extends TestCase {

  protected static final int numMaps = 5;

  protected static final int numTrees = 32;

  /** instances per partition */
  protected static final int numInstances = 20;

  public void testProcessOutput() throws Exception {
    Configuration conf = new Configuration();
    conf.setInt("mapred.map.tasks", numMaps);

    Random rng = RandomUtils.getRandom();

    // prepare the output
    TreeID[] keys = new TreeID[numTrees];
    MapredOutput[] values = new MapredOutput[numTrees];
    int[] firstIds = new int[numMaps];
    randomKeyValues(rng, keys, values, firstIds);

    // store the output in a sequence file
    Path base = new Path("testdata");
    FileSystem fs = base.getFileSystem(conf);
    if (fs.exists(base))
      fs.delete(base, true);

    Path outputFile = new Path(base, "PartialBuilderTest.seq");
    Writer writer = SequenceFile.createWriter(fs, conf, outputFile,
        TreeID.class, MapredOutput.class);

    for (int index = 0; index < numTrees; index++) {
      writer.append(keys[index], values[index]);
    }
    writer.close();

    // load the output and make sure its valid
    TreeID[] newKeys = new TreeID[numTrees];
    Node[] newTrees = new Node[numTrees];
    
    PartialBuilder.processOutput(new Job(conf), base, firstIds, newKeys, newTrees, 
        new TestCallback(keys, values));

    // check the forest
    for (int tree = 0; tree < numTrees; tree++) {
      assertEquals(values[tree].getTree(), newTrees[tree]);
    }

    assertTrue("keys not equal", Arrays.deepEquals(keys, newKeys));
  }

  /**
   * Make sure that the builder passes the good parameters to the job
   * 
   */
  public void testConfigure() {
    TreeBuilder treeBuilder = new DefaultTreeBuilder();
    Path dataPath = new Path("notUsedDataPath");
    Path datasetPath = new Path("notUsedDatasetPath");
    Long seed = 5L;

    new PartialBuilderChecker(treeBuilder, dataPath, datasetPath, seed);
  }

  /**
   * Generates random (key, value) pairs. Shuffles the partition's order
   * 
   * @param rng
   * @param keys
   * @param values
   * @param firstIds partitions's first ids in hadoop's order
   */
  protected static void randomKeyValues(Random rng, TreeID[] keys,
      MapredOutput[] values, int[] firstIds) {
    int index = 0;
    int firstId = 0;
    List<Integer> partitions = new ArrayList<Integer>();

    for (int p = 0; p < numMaps; p++) {
      // select a random partition, not yet selected
      int partition;
      do {
        partition = rng.nextInt(numMaps);
      } while (partitions.contains(partition));

      partitions.add(partition);

      int nbTrees = Step1Mapper.nbTrees(numMaps, numTrees, partition);

      for (int treeId = 0; treeId < nbTrees; treeId++) {
        Node tree = new Leaf(rng.nextInt(100));

        keys[index] = new TreeID(partition, treeId);
        values[index] = new MapredOutput(tree, nextIntArray(rng, numInstances));

        index++;
      }
      
      firstIds[p] = firstId;
      firstId += numInstances;
    }

  }

  protected static int[] nextIntArray(Random rng, int size) {
    int[] array = new int[size];
    for (int index = 0; index < size; index++) {
      array[index] = rng.nextInt(101) - 1;
    }

    return array;
  }

  protected static class PartialBuilderChecker extends PartialBuilder {

    protected final Long _seed;

    protected final TreeBuilder _treeBuilder;

    protected final Path _datasetPath;

    protected PartialBuilderChecker(TreeBuilder treeBuilder, Path dataPath,
        Path datasetPath, Long seed) {
      super(treeBuilder, dataPath, datasetPath, seed);

      _seed = seed;
      _treeBuilder = treeBuilder;
      _datasetPath = datasetPath;
    }

    @Override
    protected boolean runJob(Job job) throws IOException {
      // no need to run the job, just check if the params are correct

      Configuration conf = job.getConfiguration();
      
      assertEquals(_seed, getRandomSeed(conf));

      // PartialBuilder should detect the 'local' mode and overrides the number
      // of map tasks
      assertEquals(1, conf.getInt("mapred.map.tasks", -1));

      assertEquals(numTrees, getNbTrees(conf));

      assertFalse(isOutput(conf));
      assertTrue(isOobEstimate(conf));

      assertEquals(_treeBuilder, getTreeBuilder(conf));

      assertEquals(_datasetPath, getDistributedCacheFile(conf, 0));
      
      return true;
    }

  }

  /**
   * Mock Callback. Make sure that the callback receives the correct predictions
   * 
   */
  protected static class TestCallback implements PredictionCallback {

    protected final TreeID[] keys;

    protected final MapredOutput[] values;

    protected TestCallback(TreeID[] keys, MapredOutput[] values) {
      this.keys = keys;
      this.values = values;
    }

    @Override
    public void prediction(int treeId, int instanceId, int prediction) {
      int partition = instanceId / numInstances;

      TreeID key = new TreeID(partition, treeId);
      int index = ArrayUtils.indexOf(keys, key);
      assertTrue("key not found", index >= 0);

      assertEquals(values[index].getPredictions()[instanceId % numInstances],
          prediction);
    }

  }
}
