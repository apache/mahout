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

package org.apache.mahout.classifier.df.mapreduce.partial;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapreduce.Job;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.classifier.df.builder.DefaultTreeBuilder;
import org.apache.mahout.classifier.df.builder.TreeBuilder;
import org.apache.mahout.classifier.df.mapreduce.MapredOutput;
import org.apache.mahout.classifier.df.node.Leaf;
import org.apache.mahout.classifier.df.node.Node;
import org.junit.Test;

public final class PartialBuilderTest extends MahoutTestCase {

  private static final int NUM_MAPS = 5;

  private static final int NUM_TREES = 32;

  /** instances per partition */
  private static final int NUM_INSTANCES = 20;

  @Test
  public void testProcessOutput() throws Exception {
    Configuration conf = getConfiguration();
    conf.setInt("mapred.map.tasks", NUM_MAPS);

    Random rng = RandomUtils.getRandom();

    // prepare the output
    TreeID[] keys = new TreeID[NUM_TREES];
    MapredOutput[] values = new MapredOutput[NUM_TREES];
    int[] firstIds = new int[NUM_MAPS];
    randomKeyValues(rng, keys, values, firstIds);

    // store the output in a sequence file
    Path base = getTestTempDirPath("testdata");
    FileSystem fs = base.getFileSystem(conf);

    Path outputFile = new Path(base, "PartialBuilderTest.seq");
    Writer writer = SequenceFile.createWriter(fs, conf, outputFile,
        TreeID.class, MapredOutput.class);

    try {
      for (int index = 0; index < NUM_TREES; index++) {
        writer.append(keys[index], values[index]);
      }
    } finally {
      Closeables.close(writer, false);
    }

    // load the output and make sure its valid
    TreeID[] newKeys = new TreeID[NUM_TREES];
    Node[] newTrees = new Node[NUM_TREES];
    
    PartialBuilder.processOutput(new Job(conf), base, newKeys, newTrees);

    // check the forest
    for (int tree = 0; tree < NUM_TREES; tree++) {
      assertEquals(values[tree].getTree(), newTrees[tree]);
    }

    assertTrue("keys not equal", Arrays.deepEquals(keys, newKeys));
  }

  /**
   * Make sure that the builder passes the good parameters to the job
   * 
   */
  @Test
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
  private static void randomKeyValues(Random rng, TreeID[] keys, MapredOutput[] values, int[] firstIds) {
    int index = 0;
    int firstId = 0;
    Collection<Integer> partitions = Lists.newArrayList();

    for (int p = 0; p < NUM_MAPS; p++) {
      // select a random partition, not yet selected
      int partition;
      do {
        partition = rng.nextInt(NUM_MAPS);
      } while (partitions.contains(partition));

      partitions.add(partition);

      int nbTrees = Step1Mapper.nbTrees(NUM_MAPS, NUM_TREES, partition);

      for (int treeId = 0; treeId < nbTrees; treeId++) {
        Node tree = new Leaf(rng.nextInt(100));

        keys[index] = new TreeID(partition, treeId);
        values[index] = new MapredOutput(tree, nextIntArray(rng, NUM_INSTANCES));

        index++;
      }
      
      firstIds[p] = firstId;
      firstId += NUM_INSTANCES;
    }

  }

  private static int[] nextIntArray(Random rng, int size) {
    int[] array = new int[size];
    for (int index = 0; index < size; index++) {
      array[index] = rng.nextInt(101) - 1;
    }

    return array;
  }

  static class PartialBuilderChecker extends PartialBuilder {

    private final Long seed;

    private final TreeBuilder treeBuilder;

    private final Path datasetPath;

    PartialBuilderChecker(TreeBuilder treeBuilder, Path dataPath,
        Path datasetPath, Long seed) {
      super(treeBuilder, dataPath, datasetPath, seed);

      this.seed = seed;
      this.treeBuilder = treeBuilder;
      this.datasetPath = datasetPath;
    }

    @Override
    protected boolean runJob(Job job) throws IOException {
      // no need to run the job, just check if the params are correct

      Configuration conf = job.getConfiguration();
      
      assertEquals(seed, getRandomSeed(conf));

      // PartialBuilder should detect the 'local' mode and overrides the number
      // of map tasks
      assertEquals(1, conf.getInt("mapred.map.tasks", -1));

      assertEquals(NUM_TREES, getNbTrees(conf));

      assertFalse(isOutput(conf));

      assertEquals(treeBuilder, getTreeBuilder(conf));

      assertEquals(datasetPath, getDistributedCacheFile(conf, 0));
      
      return true;
    }

  }
}
