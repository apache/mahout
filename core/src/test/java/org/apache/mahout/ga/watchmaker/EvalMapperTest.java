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

import junit.framework.TestCase;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.ga.watchmaker.utils.DummyCandidate;
import org.apache.mahout.ga.watchmaker.utils.DummyEvaluator;
import org.apache.mahout.utils.DummyOutputCollector;
import org.apache.mahout.utils.StringUtils;
import org.uncommons.watchmaker.framework.FitnessEvaluator;

import java.util.List;
import java.util.Set;

public class EvalMapperTest extends TestCase {

  public void testMap() throws Exception {
    // population to evaluate
    int populationSize = 100;
    List<DummyCandidate> population = DummyCandidate
        .generatePopulation(populationSize);

    // fitness evaluator
    DummyEvaluator.clearEvaluations();
    FitnessEvaluator<DummyCandidate> evaluator = new DummyEvaluator();

    // Mapper
    EvalMapper mapper = new EvalMapper();
    DummyOutputCollector<LongWritable, DoubleWritable> collector = new DummyOutputCollector<LongWritable, DoubleWritable>();

    // prepare configuration
    JobConf conf = new JobConf();
    conf.set(EvalMapper.MAHOUT_GA_EVALUATOR, StringUtils.toString(evaluator));
    mapper.configure(conf);

    // evaluate the population using the mapper
    for (int index = 0; index < population.size(); index++) {
      DummyCandidate candidate = population.get(index);
      mapper.map(new LongWritable(index), new Text(StringUtils
          .toString(candidate)), collector, null);
    }

    // check that the evaluations are correct
    Set<String> keys = collector.getKeys();
    assertEquals("Number of evaluations", populationSize, keys.size());
    for (String key : keys) {
      DummyCandidate candidate = population.get(Integer.parseInt(key));
      assertEquals("Values for key " + key, 1, collector.getValue(key).size());
      double fitness = collector.getValue(key).get(0).get();
      assertEquals("Evaluation of the candidate " + key, DummyEvaluator
          .getFitness(candidate.getIndex()), fitness);
    }
  }

}
