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

package org.apache.mahout.ga.watchmaker.utils;

import org.uncommons.watchmaker.framework.FitnessEvaluator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Dummy FitnessEvaluator that stores the evaluations it calculates. Uses a static storage to handle the evaluator
 * duplication when passed as a Job parameter.
 */
public class DummyEvaluator implements FitnessEvaluator<DummyCandidate> {

  private final Random rng = new Random();

  private static final Map<Integer, Double> evaluations = new HashMap<Integer, Double>();

  public static double getFitness(Integer key) {
    if (!evaluations.containsKey(key)) {
      throw new RuntimeException("Fitness not found");
    }
    return evaluations.get(key);
  }

  public static void clearEvaluations() {
    evaluations.clear();
  }

  @Override
  public double getFitness(DummyCandidate candidate,
                           List<? extends DummyCandidate> population) {
    if (evaluations.containsKey(candidate.getIndex())) {
      throw new RuntimeException("Duplicate Fitness");
    }

    double fitness = rng.nextDouble();
    evaluations.put(candidate.getIndex(), fitness);

    return fitness;
  }

  @Override
  public boolean isNatural() {
    return false;
  }
}
