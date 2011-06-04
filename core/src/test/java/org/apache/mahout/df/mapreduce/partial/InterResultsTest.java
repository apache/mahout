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

import java.util.Random;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.data.Utils;
import org.apache.mahout.df.node.Leaf;
import org.apache.mahout.df.node.Node;
import org.junit.Before;
import org.junit.Test;

public final class InterResultsTest extends MahoutTestCase {

  /** nb attributes per generated data instance */
  private static final int NUM_ATTRIBUTES = 4;

  /** nb generated data instances */
  private static final int NUM_INSTANCES = 100;

  /** nb trees to build */
  private static final int NUM_TREES = 11;

  /** nb mappers to use */
  private static final int NUM_MAPPERS = 5;

  private String[][] splits;

  private TreeID[] keys;

  private Node[] trees;
  
  private int[] sizes;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Random rng = RandomUtils.getRandom();

    // prepare the data
    double[][] source = Utils.randomDoubles(rng, NUM_ATTRIBUTES, NUM_INSTANCES);
    String[] sData = Utils.double2String(source);

    splits = Utils.splitData(sData, NUM_MAPPERS);

    sizes = new int[NUM_MAPPERS];
    for (int p = 0; p < NUM_MAPPERS; p++) {
      sizes[p] = splits[p].length;
    }

    // prepare first step output
    keys = new TreeID[NUM_TREES];
    trees = new Node[NUM_TREES];

    int treeIndex = 0;
    for (int partition = 0; partition < NUM_MAPPERS; partition++) {
      int nbMapTrees = Step1Mapper.nbTrees(NUM_MAPPERS, NUM_TREES, partition);

      for (int index = 0; index < nbMapTrees; index++, treeIndex++) {
        keys[treeIndex] = new TreeID(partition, treeIndex);

        // put the tree index in the leaf's label
        // this way we can check the stored data
        trees[treeIndex] = new Leaf(treeIndex);
      }
    }
  }

  @Test
  public void testLoad() throws Exception {
    // store the intermediate results
    Path forestPath = new Path("testdata/InterResultsTest/test.forest");
    FileSystem fs = forestPath.getFileSystem(new Configuration());

    InterResults.store(fs, forestPath, keys, trees, sizes);

    for (int partition = 0; partition < NUM_MAPPERS; partition++) {
      int nbConcerned = Step2Mapper.nbConcerned(NUM_MAPPERS, NUM_TREES, partition);

      TreeID[] newKeys = new TreeID[nbConcerned];
      Node[] newValues = new Node[nbConcerned];

      int numInstances = InterResults.load(fs, forestPath, NUM_MAPPERS,
          NUM_TREES, partition, newKeys, newValues);

      // verify the partition's size
      assertEquals(splits[partition].length, numInstances);

      // verify (key, tree)
      int current = 0;
      for (int index = 0; index < NUM_TREES; index++) {
        // the trees of the current partition should not be loaded
        if (current < nbConcerned) {
          assertFalse("A tree from the current partition has been loaded",
              newKeys[current].partition() == partition);
        }
        if (keys[index].partition() == partition) {
          continue;
        }

        assertEquals("index: " + index, keys[index], newKeys[current]);
        assertEquals("index: " + index, trees[index], newValues[current]);

        current++;
      }
    }
  }

  @Test
  public void testStore() throws Exception {
    // store the intermediate results
    Path forestPath = new Path("testdata/InterResultsTest/test.forest");
    FileSystem fs = forestPath.getFileSystem(new Configuration());
    
    InterResults.store(fs, forestPath, keys, trees, sizes);

    // load the file and check the stored values

    FSDataInputStream in = fs.open(forestPath);

    try {
      // partitions' sizes
      for (int p = 0; p < NUM_MAPPERS; p++) {
        assertEquals(splits[p].length, in.readInt());
      }

      // load (key, tree)
      TreeID key = new TreeID();
      for (int index = 0; index < NUM_TREES; index++) {
        key.readFields(in);
        Node value = Node.read(in);

        assertEquals("index: " + index, keys[index], key);
        assertEquals("index: " + index, trees[index], value);
      }
    } finally {
      Closeables.closeQuietly(in);
    }
  }

}
