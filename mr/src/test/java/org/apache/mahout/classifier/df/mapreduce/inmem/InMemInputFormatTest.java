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

package org.apache.mahout.classifier.df.mapreduce.inmem;

import java.util.List;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.classifier.df.mapreduce.Builder;
import org.apache.mahout.classifier.df.mapreduce.inmem.InMemInputFormat.InMemInputSplit;
import org.apache.mahout.classifier.df.mapreduce.inmem.InMemInputFormat.InMemRecordReader;
import org.junit.Test;

public final class InMemInputFormatTest extends MahoutTestCase {

  @Test
  public void testSplits() throws Exception {
    int n = 1;
    int maxNumSplits = 100;
    int maxNbTrees = 1000;

    Random rng = RandomUtils.getRandom();

    for (int nloop = 0; nloop < n; nloop++) {
      int numSplits = rng.nextInt(maxNumSplits) + 1;
      int nbTrees = rng.nextInt(maxNbTrees) + 1;

      Configuration conf = getConfiguration();
      Builder.setNbTrees(conf, nbTrees);

      InMemInputFormat inputFormat = new InMemInputFormat();
      List<InputSplit> splits = inputFormat.getSplits(conf, numSplits);

      assertEquals(numSplits, splits.size());

      int nbTreesPerSplit = nbTrees / numSplits;
      int totalTrees = 0;
      int expectedId = 0;

      for (int index = 0; index < numSplits; index++) {
        assertTrue(splits.get(index) instanceof InMemInputSplit);
        
        InMemInputSplit split = (InMemInputSplit) splits.get(index);

        assertEquals(expectedId, split.getFirstId());

        if (index < numSplits - 1) {
          assertEquals(nbTreesPerSplit, split.getNbTrees());
        } else {
          assertEquals(nbTrees - totalTrees, split.getNbTrees());
        }

        totalTrees += split.getNbTrees();
        expectedId += split.getNbTrees();
      }
    }
  }

  @Test
  public void testRecordReader() throws Exception {
    int n = 1;
    int maxNumSplits = 100;
    int maxNbTrees = 1000;

    Random rng = RandomUtils.getRandom();

    for (int nloop = 0; nloop < n; nloop++) {
      int numSplits = rng.nextInt(maxNumSplits) + 1;
      int nbTrees = rng.nextInt(maxNbTrees) + 1;

      Configuration conf = getConfiguration();
      Builder.setNbTrees(conf, nbTrees);

      InMemInputFormat inputFormat = new InMemInputFormat();
      List<InputSplit> splits = inputFormat.getSplits(conf, numSplits);

      for (int index = 0; index < numSplits; index++) {
        InMemInputSplit split = (InMemInputSplit) splits.get(index);
        InMemRecordReader reader = new InMemRecordReader(split);

        reader.initialize(split, null);
        
        for (int tree = 0; tree < split.getNbTrees(); tree++) {
          // reader.next() should return true until there is no tree left
          assertEquals(tree < split.getNbTrees(), reader.nextKeyValue());
          assertEquals(split.getFirstId() + tree, reader.getCurrentKey().get());
        }
      }
    }
  }
}
