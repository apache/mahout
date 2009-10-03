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

package org.apache.mahout.df.mapred.inmem;

import java.util.Random;

import junit.framework.TestCase;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.mapred.Builder;
import org.apache.mahout.df.mapred.inmem.InMemInputFormat.InMemInputSplit;
import org.apache.mahout.df.mapred.inmem.InMemInputFormat.InMemRecordReader;

public class InMemInputFormatTest extends TestCase {

  public void testSplits() throws Exception {
    int n = 1;
    int maxNumSplits = 100;
    int maxNbTrees = 1000;

    Random rng = RandomUtils.getRandom();

    for (int nloop = 0; nloop < n; nloop++) {
      int numSplits = rng.nextInt(maxNumSplits) + 1;
      int nbTrees = rng.nextInt(maxNbTrees) + 1;

      JobConf conf = new JobConf();
      Builder.setNbTrees(conf, nbTrees);

      InMemInputFormat inputFormat = new InMemInputFormat();

      InputSplit[] splits = inputFormat.getSplits(conf, numSplits);

      assertEquals(numSplits, splits.length);

      int nbTreesPerSplit = nbTrees / numSplits;
      int totalTrees = 0;
      int expectedId = 0;

      for (int index = 0; index < numSplits; index++) {
        assertTrue(splits[index] instanceof InMemInputSplit);

        InMemInputSplit split = (InMemInputSplit) splits[index];

        assertEquals(expectedId, split.getFirstId());

        if (index < numSplits - 1)
          assertEquals(nbTreesPerSplit, split.getNbTrees());
        else
          assertEquals(nbTrees - totalTrees, split.getNbTrees());

        totalTrees += split.getNbTrees();
        expectedId += split.getNbTrees();
      }
    }
  }

  public void testRecordReader() throws Exception {
    int n = 1;
    int maxNumSplits = 100;
    int maxNbTrees = 1000;

    Random rng = RandomUtils.getRandom();

    for (int nloop = 0; nloop < n; nloop++) {
      int numSplits = rng.nextInt(maxNumSplits) + 1;
      int nbTrees = rng.nextInt(maxNbTrees) + 1;

      JobConf conf = new JobConf();
      Builder.setNbTrees(conf, nbTrees);

      InMemInputFormat inputFormat = new InMemInputFormat();

      InputSplit[] splits = inputFormat.getSplits(conf, numSplits);

      for (int index = 0; index < numSplits; index++) {
        InMemInputSplit split = (InMemInputSplit) splits[index];
        InMemRecordReader reader = (InMemRecordReader) inputFormat.getRecordReader(
            split, conf, null);

        for (int tree = 0; tree < split.getNbTrees(); tree++) {
          IntWritable key = reader.createKey();
          NullWritable value = reader.createValue();

          // reader.next() should return true until there is no tree left
          assertEquals(tree < split.getNbTrees(), reader.next(key, value));
          
          assertEquals(split.getFirstId() + tree, key.get());
        }
      }
    }
  }
}
