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
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.ga.watchmaker.utils.DummyCandidate;
import org.apache.mahout.ga.watchmaker.utils.DummyEvaluator;
import org.apache.mahout.utils.StringUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class MahoutEvaluatorTest extends TestCase {

  public <T> void testEvaluate() throws Exception {
    // candidate population
    int populationSize = 100;
    List<DummyCandidate> population = DummyCandidate
        .generatePopulation(populationSize);

    // fitness evaluator
    DummyEvaluator.clearEvaluations();
    DummyEvaluator evaluator = new DummyEvaluator();

    // run MahoutEvaluator
    List<Double> results = new ArrayList<Double>();
    MahoutEvaluator.evaluate(evaluator, population, results);

    // check results
    assertEquals("Number of evaluations", populationSize, results.size());
    for (int index = 0; index < population.size(); index++) {
      DummyCandidate candidate = population.get(index);
      assertEquals("Evaluation of the candidate " + index, DummyEvaluator
          .getFitness(candidate.getIndex()), results.get(index));
    }
  }

  public void testStoreLoadPopulation() throws Exception {
    int populationSize = 100;

    List<DummyCandidate> population = DummyCandidate
        .generatePopulation(populationSize);

    storeLoadPopulation(population);
  }

  private static void storeLoadPopulation(List<DummyCandidate> population)
      throws IOException {
    Path f = new Path("build/test.txt");
    FileSystem fs = FileSystem.get(f.toUri(), new Configuration());

    // store the population
    MahoutEvaluator.storePopulation(fs, f, population);

    // load the population
    List<DummyCandidate> inpop = new ArrayList<DummyCandidate>();
    loadPopulation(fs, f, inpop);

    // check that the file contains the correct population
    assertEquals("Population size", population.size(), inpop.size());
    for (int index = 0; index < population.size(); index++) {
      assertEquals("Bad candidate " + index, population.get(index), inpop
          .get(index));
    }
  }

  private static void loadPopulation(FileSystem fs, Path f,
                                     List<DummyCandidate> population) throws IOException {
    FSDataInputStream in = fs.open(f);
    BufferedReader reader = new BufferedReader(new InputStreamReader(in));

    try {
      String s;
      while ((s = reader.readLine()) != null) {
        population.add((DummyCandidate) StringUtils.fromString(s));
      }
    } finally {
      reader.close();
    }
  }

}
