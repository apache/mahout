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

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.examples.MahoutTestCase;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.Rule;
import org.apache.mahout.ga.watchmaker.cd.utils.RandomRule;
import org.apache.mahout.ga.watchmaker.cd.utils.RandomRuleResults;

import com.google.common.io.Resources;
import org.junit.Test;

public final class CDMahoutEvaluatorTest extends MahoutTestCase {

  @Test
  public void testEvaluate() throws Exception {
    int nbrules = 100;
    Random rng = RandomUtils.getRandom();
    int target = 1;

    // random rules
    List<Rule> rules = Lists.newArrayList();
    for (int index = 0; index < nbrules; index++) {
      rules.add(new RandomRule(index, target, rng));
    }

    // dataset
    // This is sensitive to the working directory where the test is run:
    FileSystem fs = FileSystem.get(new Configuration());
    Path input = fs.makeQualified(new Path(Resources.getResource("wdbc").toString()));
    CDMahoutEvaluator.initializeDataSet(input);

    // evaluate the rules
    List<CDFitness> results = Lists.newArrayList();
    Path output = getTestTempDirPath("output");
    fs = output.getFileSystem(new Configuration());
    fs.delete(output, true); // It's unhappy if this directory exists
    CDMahoutEvaluator.evaluate(rules, target, input, output, results);

    // check the results
    for (int index = 0; index < nbrules; index++) {
      assertEquals("rule " + index, RandomRuleResults.getResult(index),
          results.get(index));
    }

  }

}
