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

package org.apache.mahout.ga.watchmaker.travellingsalesman;

import org.apache.mahout.ga.watchmaker.MahoutFitnessEvaluator;
import org.uncommons.maths.random.MersenneTwisterRNG;
import org.uncommons.maths.random.PoissonGenerator;
import org.uncommons.watchmaker.framework.CandidateFactory;
import org.uncommons.watchmaker.framework.ConcurrentEvolutionEngine;
import org.uncommons.watchmaker.framework.EvolutionEngine;
import org.uncommons.watchmaker.framework.EvolutionObserver;
import org.uncommons.watchmaker.framework.EvolutionaryOperator;
import org.uncommons.watchmaker.framework.FitnessEvaluator;
import org.uncommons.watchmaker.framework.PopulationData;
import org.uncommons.watchmaker.framework.SelectionStrategy;
import org.uncommons.watchmaker.framework.SequentialEvolutionEngine;
import org.uncommons.watchmaker.framework.factories.ListPermutationFactory;
import org.uncommons.watchmaker.framework.operators.EvolutionPipeline;
import org.uncommons.watchmaker.framework.operators.ListOrderCrossover;
import org.uncommons.watchmaker.framework.operators.ListOrderMutation;
import org.uncommons.watchmaker.framework.termination.GenerationCount;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * Evolutionary algorithm for finding (approximate) solutions to the travelling
 * salesman problem.
 * 
 * 
 * <br>
 * The original code is from <b>the Watchmaker project</b>
 * (https://watchmaker.dev.java.net/).<br>
 * Modified to use Mahout whenever requested.
 */
public class EvolutionaryTravellingSalesman implements
    TravellingSalesmanStrategy {
  private final DistanceLookup distances;

  private final SelectionStrategy<? super List<String>> selectionStrategy;

  private final int populationSize;

  private final int eliteCount;

  private final int generationCount;

  private final boolean crossover;

  private final boolean mutation;

  private final boolean mahout;

  /**
   * Creates an evolutionary Travelling Salesman solver with the specified
   * configuration.
   * 
   * @param distances Information about the distances between cities.
   * @param selectionStrategy The selection implementation to use for the
   *        evolutionary algorithm.
   * @param populationSize The number of candidates in the population of evolved
   *        routes.
   * @param eliteCount The number of candidates to preserve via elitism at each
   *        generation.
   * @param generationCount The number of iterations of evolution to perform.
   * @param crossover Whether or not to use a cross-over operator in the
   *        evolution.
   * @param mutation Whether or not to use a mutation operator in the evolution.
   * @param mahout Whether or not to use Mahout for evaluation.
   */
  public EvolutionaryTravellingSalesman(DistanceLookup distances,
      SelectionStrategy<? super List<String>> selectionStrategy,
      int populationSize, int eliteCount, int generationCount,
      boolean crossover, boolean mutation, boolean mahout) {
    if (eliteCount < 0 || eliteCount >= populationSize) {
      throw new IllegalArgumentException(
          "Elite count must be non-zero and less than population size.");
    }
    if (!crossover && !mutation) {
      throw new IllegalArgumentException(
          "At least one of cross-over or mutation must be selected.");
    }
    this.distances = distances;
    this.selectionStrategy = selectionStrategy;
    this.populationSize = populationSize;
    this.eliteCount = eliteCount;
    this.generationCount = generationCount;
    this.crossover = crossover;
    this.mutation = mutation;
    this.mahout = mahout;
  }

  @Override
  public String getDescription() {
    String selectionName = selectionStrategy.getClass().getSimpleName();
    return (mahout ? "Mahout " : "") + "Evolution (pop: " + populationSize
        + ", gen: " + generationCount + ", elite: " + eliteCount + ", "
        + selectionName + ')';
  }

  /**
   * Calculates the shortest route using a generational evolutionary algorithm
   * with a single ordered mutation operator and truncation selection.
   * 
   * @param cities The list of destinations, each of which must be visited once.
   * @param progressListener Call-back for receiving the status of the algorithm
   *        as it progresses.
   * @return The (approximate) shortest route that visits each of the specified
   *         cities once.
   */
  @Override
  public List<String> calculateShortestRoute(Collection<String> cities,
      final ProgressListener progressListener) {
    Random rng = new MersenneTwisterRNG();

    // Set-up evolution pipeline (cross-over followed by mutation).
    List<EvolutionaryOperator<? super List<?>>> operators = new ArrayList<EvolutionaryOperator<? super List<?>>>(
        2);
    if (crossover) {
      operators.add(new ListOrderCrossover());
    }
    if (mutation) {
      operators.add(new ListOrderMutation(new PoissonGenerator(1.5, rng),
          new PoissonGenerator(1.5, rng)));
    }

    EvolutionaryOperator<List<?>> pipeline = new EvolutionPipeline<List<?>>(
        operators);

    CandidateFactory<List<String>> candidateFactory = new ListPermutationFactory<String>(
        new LinkedList<String>(cities));
    EvolutionEngine<List<String>> engine = getEngine(candidateFactory,
        pipeline, rng);
    engine.addEvolutionObserver(new EvolutionObserver<List<String>>() {
      @Override
      public void populationUpdate(PopulationData<? extends List<String>> data) {
        if (progressListener != null) {
          progressListener
              .updateProgress(((double) data.getGenerationNumber() + 1)
                  / generationCount * 100);
        }
      }
    });
    return engine.evolve(populationSize, eliteCount, new GenerationCount(
        generationCount));
  }

  private EvolutionEngine<List<String>> getEngine(
      CandidateFactory<List<String>> candidateFactory,
      EvolutionaryOperator<List<?>> pipeline, Random rng) {
    if (mahout) {
      // This is what we need to do to distribute the fitness evaluation.
      // First create a STFitnessEvaluator that wraps our FitnessEvaluator
      FitnessEvaluator<List<String>> evaluator = new MahoutFitnessEvaluator<List<String>>(
          new RouteEvaluator(distances));
      // Then use a SequentialEvolutionEngine instead of a StandaloneEvolutionEngine.
      // Its parameters remain the same.
      return new SequentialEvolutionEngine<List<String>>(candidateFactory, pipeline,
          evaluator, selectionStrategy, rng);
    } else {
      return new ConcurrentEvolutionEngine<List<String>>(candidateFactory,
          pipeline, new RouteEvaluator(distances), selectionStrategy, rng);
    }
  }
}
