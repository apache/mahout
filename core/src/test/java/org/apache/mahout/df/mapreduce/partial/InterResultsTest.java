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

import junit.framework.TestCase;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.data.Utils;
import org.apache.mahout.df.node.Leaf;
import org.apache.mahout.df.node.Node;

public class InterResultsTest extends TestCase {

  /** nb attributes per generated data instance */
  protected static final int nbAttributes = 4;

  /** nb generated data instances */
  protected static final int nbInstances = 100;

  /** nb trees to build */
  protected static final int nbTrees = 11;

  /** nb mappers to use */
  protected static final int nbMappers = 5;

  protected String[][] splits;

  TreeID[] keys;

  Node[] trees;
  
  int[] sizes;

  @Override
  protected void setUp() throws Exception {
    RandomUtils.useTestSeed();
    Random rng = RandomUtils.getRandom();

    // prepare the data
    double[][] source = Utils.randomDoubles(rng, nbAttributes, nbInstances);
    String[] sData = Utils.double2String(source);

    splits = Utils.splitData(sData, nbMappers);

    sizes = new int[nbMappers];
    for (int p = 0; p < nbMappers; p++) {
      sizes[p] = splits[p].length;
    }

    // prepare first step output
    keys = new TreeID[nbTrees];
    trees = new Node[nbTrees];
    
    int treeIndex = 0;
    for (int partition = 0; partition < nbMappers; partition++) {
      int nbMapTrees = Step1Mapper.nbTrees(nbMappers, nbTrees, partition);

      for (int index = 0; index < nbMapTrees; index++, treeIndex++) {
        keys[treeIndex] = new TreeID(partition, treeIndex);

        // put the tree index in the leaf's label
        // this way we can check the stored data
        trees[treeIndex] = new Leaf(treeIndex);
      }
    }
  }

  public void testLoad() throws Exception {
    // store the intermediate results
    Path forestPath = new Path("testdata/InterResultsTest/test.forest");
    FileSystem fs = forestPath.getFileSystem(new Configuration());

    InterResults.store(fs, forestPath, keys, trees, sizes);

    for (int partition = 0; partition < nbMappers; partition++) {
      int nbConcerned = Step2Mapper.nbConcerned(nbMappers, nbTrees, partition);

      TreeID[] newKeys = new TreeID[nbConcerned];
      Node[] newValues = new Node[nbConcerned];

      int numInstances = InterResults.load(fs, forestPath, nbMappers,
          nbTrees, partition, newKeys, newValues);

      // verify the partition's size
      assertEquals(splits[partition].length, numInstances);

      // verify (key, tree)
      int current = 0;
      for (int index = 0; index < nbTrees; index++) {
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

  public void testStore() throws Exception {
    // store the intermediate results
    Path forestPath = new Path("testdata/InterResultsTest/test.forest");
    FileSystem fs = forestPath.getFileSystem(new Configuration());
    
    InterResults.store(fs, forestPath, keys, trees, sizes);

    // load the file and check the stored values

    FSDataInputStream in = fs.open(forestPath);

    try {
      // partitions' sizes
      for (int p = 0; p < nbMappers; p++) {
        assertEquals(splits[p].length, in.readInt());
      }

      // load (key, tree)
      TreeID key = new TreeID();
      for (int index = 0; index < nbTrees; index++) {
        key.readFields(in);
        Node value = Node.read(in);

        assertEquals("index: " + index, keys[index], key);
        assertEquals("index: " + index, trees[index], value);
      }
    } finally {
      in.close();
    }
  }

}
