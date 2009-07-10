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

import java.io.IOException;
import java.util.List;

/** Watchmaker compatible Fitness Evaluator that delegates the evaluation of the whole population to Mahout. */
public class MahoutFitnessEvaluator<T> extends STFitnessEvaluator<T> {

  private final FitnessEvaluator<? super T> evaluator;

  public MahoutFitnessEvaluator(FitnessEvaluator<? super T> evaluator) {
    this.evaluator = evaluator;
  }

  @Override
  protected void evaluate(List<? extends T> population, List<Double> evaluations) {
    try {
      MahoutEvaluator.evaluate(evaluator, population, evaluations);
    } catch (IOException e) {
      throw new RuntimeException("Exception while evaluating the population", e);
    }
  }

  @Override
  public boolean isNatural() {
    return evaluator.isNatural();
  }

}
