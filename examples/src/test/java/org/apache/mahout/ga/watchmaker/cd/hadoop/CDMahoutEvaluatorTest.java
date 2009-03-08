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
import org.apache.hadoop.fs.Path;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.Rule;
import org.apache.mahout.ga.watchmaker.cd.utils.RandomRule;
import org.apache.mahout.ga.watchmaker.cd.utils.RandomRuleResults;
import org.uncommons.maths.random.MersenneTwisterRNG;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class CDMahoutEvaluatorTest extends TestCase {

  public void testEvaluate() throws Exception {
    int nbrules = 100;
    Random rng = new MersenneTwisterRNG();
    int target = 1;

    // random rules
    List<Rule> rules = new ArrayList<Rule>();
    for (int index = 0; index < nbrules; index++) {
      rules.add(new RandomRule(index, target, rng));
    }

    // dataset
    Path input = new Path("target/test-classes/wdbc");
    CDMahoutEvaluator.initializeDataSet(input);

    // evaluate the rules
    List<CDFitness> results = new ArrayList<CDFitness>();
    CDMahoutEvaluator.evaluate(rules, target, input, results);

    // check the results
    for (int index = 0; index < nbrules; index++) {
      assertEquals("rule " + index, RandomRuleResults.getResult(index),
          results.get(index));
    }

  }

}
