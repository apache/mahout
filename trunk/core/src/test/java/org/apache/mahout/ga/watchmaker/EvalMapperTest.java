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

package org.apache.mahout.ga.watchmaker;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.ga.watchmaker.utils.DummyCandidate;
import org.apache.mahout.ga.watchmaker.utils.DummyEvaluator;
import org.apache.mahout.common.StringUtils;
import org.junit.Test;
import org.uncommons.watchmaker.framework.FitnessEvaluator;

import java.util.List;
import java.util.Set;

public final class EvalMapperTest extends MahoutTestCase {

  private static final int POPULATION_SIZE = 100;

  @Test
  public void testMap() throws Exception {
    // population to evaluate
    List<DummyCandidate> population = DummyCandidate
        .generatePopulation(POPULATION_SIZE);

    // fitness evaluator
    DummyEvaluator.clearEvaluations();
    FitnessEvaluator<DummyCandidate> evaluator = new DummyEvaluator();

    // Mapper
    EvalMapper mapper = new EvalMapper();
    Configuration conf = new Configuration();
    conf.set(EvalMapper.MAHOUT_GA_EVALUATOR, StringUtils.toString(evaluator));
    DummyRecordWriter<LongWritable,DoubleWritable> output = new DummyRecordWriter<LongWritable,DoubleWritable>();
    Mapper<LongWritable,Text,LongWritable,DoubleWritable>.Context context =
        DummyRecordWriter.build(mapper, conf, output);

    mapper.setup(context);

    // evaluate the population using the mapper
    for (int index = 0; index < population.size(); index++) {
      DummyCandidate candidate = population.get(index);
      mapper.map(new LongWritable(index), new Text(StringUtils.toString(candidate)), context);
    }

    // check that the evaluations are correct
    Set<LongWritable> keys = output.getKeys();
    assertEquals("Number of evaluations", POPULATION_SIZE, keys.size());
    for (LongWritable key : keys) {
      DummyCandidate candidate = population.get((int) key.get());
      assertEquals("Values for key " + key, 1, output.getValue(key).size());
      double fitness = output.getValue(key).get(0).get();
      assertEquals("Evaluation of the candidate " + key,
                   DummyEvaluator.getFitness(candidate.getIndex()),
                   fitness,
                   EPSILON);
    }
  }

}
