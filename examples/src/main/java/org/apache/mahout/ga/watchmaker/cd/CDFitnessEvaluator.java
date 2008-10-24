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

package org.apache.mahout.ga.watchmaker.cd;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.ga.watchmaker.STFitnessEvaluator;
import org.apache.mahout.ga.watchmaker.cd.hadoop.CDMahoutEvaluator;
import org.apache.mahout.ga.watchmaker.cd.hadoop.DatasetSplit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Class Discovery Fitness Evaluator. Delegates to Mahout the task of evaluating
 * the fitness.
 */
public class CDFitnessEvaluator extends STFitnessEvaluator<Rule> {

  private final Path dataset;

  private final DatasetSplit split;

  private final List<CDFitness> evals = new ArrayList<CDFitness>();
  
  private final int target;

  /**
   * 
   * @param dataset dataset path
   * @param split
   */
  public CDFitnessEvaluator(String dataset, int target, DatasetSplit split) {
    this.dataset = new Path(dataset);
    this.target = target;
    this.split = split;
  }

  @Override
  public boolean isNatural() {
    return true;
  }

  @Override
  protected void evaluate(List<? extends Rule> population,
      List<Double> evaluations) {
    evals.clear();

    try {
      CDMahoutEvaluator.evaluate(population, target, dataset, evals, split);
    } catch (IOException e) {
      throw new RuntimeException("Exception while evaluating the population", e);
    }

    for (CDFitness fitness : evals)
      evaluations.add(fitness.get());
  }

}
