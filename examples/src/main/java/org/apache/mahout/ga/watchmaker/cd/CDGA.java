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
import org.apache.mahout.ga.watchmaker.cd.hadoop.CDMahoutEvaluator;
import org.apache.mahout.ga.watchmaker.cd.hadoop.DatasetSplit;
import org.uncommons.maths.random.MersenneTwisterRNG;
import org.uncommons.watchmaker.framework.CandidateFactory;
import org.uncommons.watchmaker.framework.EvolutionEngine;
import org.uncommons.watchmaker.framework.EvolutionObserver;
import org.uncommons.watchmaker.framework.EvolutionaryOperator;
import org.uncommons.watchmaker.framework.FitnessEvaluator;
import org.uncommons.watchmaker.framework.PopulationData;
import org.uncommons.watchmaker.framework.SelectionStrategy;
import org.uncommons.watchmaker.framework.SequentialEvolutionEngine;
import org.uncommons.watchmaker.framework.operators.EvolutionPipeline;
import org.uncommons.watchmaker.framework.selection.RouletteWheelSelection;
import org.uncommons.watchmaker.framework.termination.GenerationCount;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Class Discovery Genetic Algorithm main class. Has the following parameters:
 * <ul>
 * <li>threshold<br>
 * Condition activation threshold. See Also
 * {@link org.apache.mahout.ga.watchmaker.cd.CDRule CDRule}
 * <li>nb cross point<br>
 * Number of points used by the{@link org.apache.mahout.ga.watchmaker.cd.CDCrossover CrossOver}
 * operator
 * <li>mutation rate<br>
 * mutation rate of the
 * {@link org.apache.mahout.ga.watchmaker.cd.CDMutation Mutation} operator
 * <li>mutation range<br>
 * mutation range of the
 * {@link org.apache.mahout.ga.watchmaker.cd.CDMutation Mutation} operator
 * <li>mutation precision<br>
 * mutation precision of the
 * {@link org.apache.mahout.ga.watchmaker.cd.CDMutation Mutation} operator
 * <li>population size
 * <li>generations count<br>
 * number of generations the genetic algorithm will be run for.
 * 
 * </ul>
 */
public class CDGA {

  private static final Logger log = LoggerFactory.getLogger(CDGA.class);

  private CDGA() {
  }

  public static void main(String[] args) throws IOException {
    String dataset = "target/classes/wdbc";
    int target = 1;
    double threshold = 0.5;
    int crosspnts = 1;
    double mutrate = 0.1;
    double mutrange = 0.1; // 10%
    int mutprec = 2;
    int popSize = 10;
    int genCount = 10;

    if (args.length == 9) {
      dataset = args[0];
      target = Integer.parseInt(args[1]);
      threshold = Double.parseDouble(args[2]);
      crosspnts = Integer.parseInt(args[3]);
      mutrate = Double.parseDouble(args[4]);
      mutrange = Double.parseDouble(args[5]);
      mutprec = Integer.parseInt(args[6]);
      popSize = Integer.parseInt(args[7]);
      genCount = Integer.parseInt(args[8]);
    } else {
      log.warn("Invalid arguments, working with default parameters instead");
	  }

    long start = System.currentTimeMillis();

    runJob(dataset, target, threshold, crosspnts, mutrate, mutrange, mutprec,
        popSize, genCount);

    long end = System.currentTimeMillis();

    printElapsedTime(end - start);
  }

  private static void runJob(String dataset, int target, double threshold,
      int crosspnts, double mutrate, double mutrange, int mutprec, int popSize,
      int genCount) throws IOException {
    Path inpath = new Path(dataset);
    CDMahoutEvaluator.initializeDataSet(inpath);

    // Candidate Factory
    CandidateFactory<CDRule> factory = new CDFactory(threshold);

    // Evolution Scheme
    List<EvolutionaryOperator<? super CDRule>> operators = new ArrayList<EvolutionaryOperator<? super CDRule>>();
    operators.add(new CDCrossover(crosspnts));
    operators.add(new CDMutation(mutrate, mutrange, mutprec));
    EvolutionPipeline<CDRule> pipeline = new EvolutionPipeline<CDRule>(operators);

    // 75 % of the dataset is dedicated to training
    DatasetSplit split = new DatasetSplit(0.75);

    // Fitness Evaluator (defaults to training)
    FitnessEvaluator<? super CDRule> evaluator = new CDFitnessEvaluator(
        dataset, target, split);
    // Selection Strategy
    SelectionStrategy<? super CDRule> selection = new RouletteWheelSelection();

    EvolutionEngine<CDRule> engine = new SequentialEvolutionEngine<CDRule>(factory,
        pipeline, evaluator, selection, new MersenneTwisterRNG());

    engine.addEvolutionObserver(new EvolutionObserver<CDRule>() {
      @Override
      public void populationUpdate(PopulationData<? extends CDRule> data) {
        log.info("Generation {}", data.getGenerationNumber());
      }
    });

    // evolve the rules over the training set
    Rule solution = engine.evolve(popSize, 1, new GenerationCount(genCount));

    // fitness over the training set
    CDFitness bestTrainFit = CDMahoutEvaluator.evaluate(solution, target,
        inpath, split);

    // fitness over the testing set
    split.setTraining(false);
    CDFitness bestTestFit = CDMahoutEvaluator.evaluate(solution, target,
        inpath, split);

    // evaluate the solution over the testing set
    log.info("Best solution fitness (train set) : {}", bestTrainFit);
    log.info("Best solution fitness (test set) : {}", bestTestFit);
  }

  private static void printElapsedTime(long milli) {
    long seconds = milli / 1000;
    milli %= 1000;

    long minutes = seconds / 60;
    seconds %= 60;

    long hours = minutes / 60;
    minutes %= 60;

    log.info("Elapsed time (Hours:minutes:seconds:milli) : {}:{}:{}:{}", new Object[] {hours, minutes, seconds, milli});
  }
}
