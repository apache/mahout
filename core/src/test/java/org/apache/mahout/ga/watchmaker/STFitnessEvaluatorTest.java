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
import org.uncommons.maths.random.MersenneTwisterRNG;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class STFitnessEvaluatorTest extends TestCase {

  private static class STFitnessEvaluatorMock<T> extends STFitnessEvaluator<T> {
    private int nbcalls;

    private List<Double> evaluations;

    public void shouldReturn(List<Double> evaluations) {
      this.evaluations = evaluations;
    }

    public int getNbCalls() {
      return nbcalls;
    }

    @Override
    protected void evaluate(List<? extends T> population,
                            List<Double> evaluations) {
      nbcalls++;
      evaluations.addAll(this.evaluations);
    }

    @Override
    public boolean isNatural() {
      // Doesn't matter
      return false;
    }

  }

  /**
   * Test method for {@link org.apache.mahout.ga.watchmaker.STFitnessEvaluator#evaluate(java.util.List,
   * java.util.List)}.<br> <br> Make sure that evaluate() is not called twice for the same population.
   */
  public void testEvaluateSamePopulation() {
    STFitnessEvaluatorMock<Integer> mock = new STFitnessEvaluatorMock<Integer>();
    Random rng = new MersenneTwisterRNG();

    int size = 100;
    List<Integer> population = randomInts(size, rng);

    List<Double> evaluations = randomFloats(size, rng);
    mock.shouldReturn(evaluations);

    for (int index = 0; index < size; index++) {
      Integer candidate = population.get(index);
      assertEquals(evaluations.get(index), mock.getFitness(candidate, population));
    }

    // getFitness() should be called once
    assertEquals(1, mock.getNbCalls());
  }

  /**
   * Test method for {@link org.apache.mahout.ga.watchmaker.STFitnessEvaluator#evaluate(java.util.List,
   * java.util.List)}.<br> <br> Make sure that evaluate() is called as many different populations are passed to
   * getFitness().
   */
  public void testEvaluateDifferentPopulations() {
    STFitnessEvaluatorMock<Integer> mock = new STFitnessEvaluatorMock<Integer>();
    Random rng = new MersenneTwisterRNG();

    // generate a population A
    int size = 100;
    List<Integer> population = randomInts(size, rng);

    List<Double> evaluations = randomFloats(size, rng);
    mock.shouldReturn(evaluations);

    // call with population A
    mock.getFitness(population.get(rng.nextInt(size)), population);

    // generate a new population B
    population = randomInts(size, rng);

    // call with population B
    mock.getFitness(population.get(rng.nextInt(size)), population);

    // getFitness() should be called twice
    assertEquals(2, mock.getNbCalls());
  }

  private static List<Integer> randomInts(int size, Random rng) {
    List<Integer> population = new ArrayList<Integer>();
    for (int index = 0; index < size; index++) {
      population.add(rng.nextInt());
    }

    return population;
  }

  private static List<Double> randomFloats(int size, Random rng) {
    List<Double> population = new ArrayList<Double>();
    for (int index = 0; index < size; index++) {
      population.add(rng.nextDouble());
    }

    return population;
  }

}
