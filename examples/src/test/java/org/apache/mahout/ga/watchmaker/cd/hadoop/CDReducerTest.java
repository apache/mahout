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

import java.util.List;
import java.util.Random;
import java.util.Set;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.examples.MahoutTestCase;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.junit.Before;
import org.junit.Test;

public final class CDReducerTest extends MahoutTestCase {

  private static final int NUM_EVALS = 100;

  private List<CDFitness> evaluations;
  private CDFitness expected;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    // generate random evaluatons and calculate expectations
    evaluations = Lists.newArrayList();
    Random rng = RandomUtils.getRandom();
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    for (int index = 0; index < NUM_EVALS; index++) {
      CDFitness fitness = new CDFitness(rng.nextInt(100), rng.nextInt(100), rng.nextInt(100), rng.nextInt(100));
      tp += fitness.getTp();
      fp += fitness.getFp();
      tn += fitness.getTn();
      fn += fitness.getFn();

      evaluations.add(fitness);
    }
    expected = new CDFitness(tp, fp, tn, fn);
  }

  @Test
  public void testReduce() throws Exception {
    CDReducer reducer = new CDReducer();
    Configuration conf = new Configuration();
    DummyRecordWriter<LongWritable, CDFitness> reduceWriter = new DummyRecordWriter<LongWritable, CDFitness>();
    Reducer<LongWritable, CDFitness, LongWritable, CDFitness>.Context reduceContext =
        DummyRecordWriter.build(reducer, conf, reduceWriter, LongWritable.class, CDFitness.class);

    LongWritable zero = new LongWritable(0);
    reducer.reduce(zero, evaluations, reduceContext);

    // check if the expectations are met
    Set<LongWritable> keys = reduceWriter.getKeys();
    assertEquals("nb keys", 1, keys.size());
    assertTrue("bad key", keys.contains(zero));

    assertEquals("nb values", 1, reduceWriter.getValue(zero).size());
    CDFitness fitness = reduceWriter.getValue(zero).get(0);
    assertEquals(expected, fitness);

  }

}
