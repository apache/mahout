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

import junit.framework.TestCase;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.mahout.ga.watchmaker.cd.hadoop.DatasetSplit.RndLineRecordReader;
import org.uncommons.maths.random.MersenneTwisterRNG;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * 
 */
public class DatasetSplitTest extends TestCase {

  /**
   * Mock RecordReader that returns a sequence of keys in the range [0, size[
   */
  private static class MockReader implements RecordReader<LongWritable, Text> {

    private long current;

    private long size;

    public MockReader(long size) {
      assert size > 0 : "size == 0";

      this.size = size;
    }

    public void close() throws IOException {
      // TODO Auto-generated method stub

    }

    public LongWritable createKey() {
      // TODO Auto-generated method stub
      return null;
    }

    public Text createValue() {
      // TODO Auto-generated method stub
      return null;
    }

    public long getPos() throws IOException {
      // TODO Auto-generated method stub
      return 0;
    }

    public float getProgress() throws IOException {
      // TODO Auto-generated method stub
      return 0;
    }

    public boolean next(LongWritable key, Text value) throws IOException {
      if (current == size) {
        return false;
      } else {
        key.set(current++);
        return true;
      }
    }
  }

  public void testTrainingTestingSets() throws IOException {
    int n = 20;

    for (int nloop = 0; nloop < n; nloop++) {
      long datasetSize = 100;
      MersenneTwisterRNG rng = new MersenneTwisterRNG();
      byte[] seed = rng.getSeed();
      double threshold = rng.nextDouble();

      JobConf conf = new JobConf();
      RndLineRecordReader rndReader;
      Set<Long> dataset = new HashSet<Long>();
      LongWritable key = new LongWritable();
      Text value = new Text();
      
      DatasetSplit split = new DatasetSplit(seed, threshold);

      // read the training set
      split.storeJobParameters(conf);
      rndReader = new RndLineRecordReader(new MockReader(datasetSize), conf);
      while (rndReader.next(key, value)) {
        assertTrue("duplicate line index", dataset.add(key.get()));
      }

      // read the testing set
      split.setTraining(false);
      split.storeJobParameters(conf);
      rndReader = new RndLineRecordReader(new MockReader(datasetSize), conf);
      while (rndReader.next(key, value)) {
        assertTrue("duplicate line index", dataset.add(key.get()));
      }

      assertEquals("missing datas", datasetSize, dataset.size());
    }
  }

  public void testStoreJobParameters() {
    int n = 20;

    for (int nloop = 0; nloop < n; nloop++) {
      MersenneTwisterRNG rng = new MersenneTwisterRNG();

      byte[] seed = rng.getSeed();
      double threshold = rng.nextDouble();
      boolean training = rng.nextBoolean();

      DatasetSplit split = new DatasetSplit(seed, threshold);
      split.setTraining(training);

      JobConf conf = new JobConf();
      split.storeJobParameters(conf);

      assertTrue("bad seed", Arrays.equals(seed, DatasetSplit.getSeed(conf)));
      assertEquals("bad threshold", threshold, DatasetSplit.getThreshold(conf));
      assertEquals("bad training", training, DatasetSplit.isTraining(conf));
    }
  }
}
