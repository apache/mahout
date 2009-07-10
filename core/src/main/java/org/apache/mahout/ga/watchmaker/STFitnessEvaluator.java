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

import org.uncommons.watchmaker.framework.FitnessEvaluator;

import java.util.ArrayList;
import java.util.List;

/** Special Fitness Evaluator that evaluates all the population ones. */
public abstract class STFitnessEvaluator<T> implements FitnessEvaluator<T> {

  private final List<Double> evaluations = new ArrayList<Double>();

  private List<? extends T> population;

  @Override
  public double getFitness(T candidate, List<? extends T> population) {
    // evaluate the population, when needed
    if (this.population == null || this.population != population) {
      evaluations.clear();
      evaluate(population, evaluations);
      this.population = population;
    }

    int index = population.indexOf(candidate);
    if (index == -1) {
      throw new RuntimeException("Candidate is not part of the population");
    }

    return evaluations.get(index);
  }

  protected abstract void evaluate(List<? extends T> population, List<Double> evaluations);

}
