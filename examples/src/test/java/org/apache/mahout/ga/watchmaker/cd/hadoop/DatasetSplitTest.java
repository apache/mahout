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

package org.apache.mahout.ga.watchmaker.cd.hadoop;

import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;
import org.apache.mahout.examples.MahoutTestCase;
import org.apache.mahout.ga.watchmaker.cd.hadoop.DatasetSplit.RndLineRecordReader;
import org.junit.Test;

public final class DatasetSplitTest extends MahoutTestCase {

  /**
   * Mock RecordReader that returns a sequence of keys in the range [0, size[
   */
  private static class MockReader extends RecordReader<LongWritable, Text> {

    private long current;
    private final long size;
    private final LongWritable currentKey = new LongWritable();
    private final Text currentValue = new Text();

    MockReader(long size) {
      Preconditions.checkArgument(size > 0, "Size must be positive", size);
      this.size = size;
    }

    @Override
    public void close() throws IOException {
    }

    @Override
    public float getProgress() throws IOException {
      return 0;
    }

    @Override
    public LongWritable getCurrentKey() throws IOException, InterruptedException {
      return currentKey;
    }

    @Override
    public Text getCurrentValue() throws IOException, InterruptedException {
      return currentValue;
    }

    @Override
    public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
      if (current == size) {
        return false;
      } else {
        currentKey.set(current++);
        return true;
      }
    }
  }

  @Test
  public void testTrainingTestingSets() throws Exception {
    int n = 20;

    for (int nloop = 0; nloop < n; nloop++) {
      RandomWrapper rng = (RandomWrapper) RandomUtils.getRandom();
      double threshold = rng.nextDouble();

      Configuration conf = new Configuration();
      Collection<Long> dataset = new HashSet<Long>();

      DatasetSplit split = new DatasetSplit(rng.getSeed(), threshold);

      // read the training set
      split.storeJobParameters(conf);
      long datasetSize = 100;
      RndLineRecordReader rndReader = new RndLineRecordReader(new MockReader(datasetSize), conf);
      while (rndReader.nextKeyValue()) {
        assertTrue("duplicate line index", dataset.add(rndReader.getCurrentKey().get()));
      }

      // read the testing set
      split.setTraining(false);
      split.storeJobParameters(conf);
      rndReader = new RndLineRecordReader(new MockReader(datasetSize), conf);
      while (rndReader.nextKeyValue()) {
        assertTrue("duplicate line index", dataset.add(rndReader.getCurrentKey().get()));
      }

      assertEquals("missing datas", datasetSize, dataset.size());
    }
  }

  @Test
  public void testStoreJobParameters() {
    int n = 20;

    for (int nloop = 0; nloop < n; nloop++) {
      RandomWrapper rng = (RandomWrapper) RandomUtils.getRandom();

      long seed = rng.getSeed();
      double threshold = rng.nextDouble();
      boolean training = rng.nextBoolean();

      DatasetSplit split = new DatasetSplit(seed, threshold);
      split.setTraining(training);

      Configuration conf = new Configuration();
      split.storeJobParameters(conf);

      assertEquals("bad seed", seed, DatasetSplit.getSeed(conf));
      assertEquals("bad threshold", threshold, DatasetSplit.getThreshold(conf), EPSILON);
      assertEquals("bad training", training, DatasetSplit.isTraining(conf));
    }
  }
}
