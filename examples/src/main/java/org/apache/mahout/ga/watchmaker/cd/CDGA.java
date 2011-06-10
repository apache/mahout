/*
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

import java.io.IOException;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.ga.watchmaker.cd.hadoop.CDMahoutEvaluator;
import org.apache.mahout.ga.watchmaker.cd.hadoop.DatasetSplit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

/**
 * Class Discovery Genetic Algorithm main class. Has the following parameters:
 * <ul>
 * <li>threshold<br>
 * Condition activation threshold. See Also {@link org.apache.mahout.ga.watchmaker.cd.CDRule CDRule}
 * <li>nb cross point<br>
 * Number of points used by the{@link org.apache.mahout.ga.watchmaker.cd.CDCrossover CrossOver} operator
 * <li>mutation rate<br>
 * mutation rate of the {@link org.apache.mahout.ga.watchmaker.cd.CDMutation Mutation} operator
 * <li>mutation range<br>
 * mutation range of the {@link org.apache.mahout.ga.watchmaker.cd.CDMutation Mutation} operator
 * <li>mutation precision<br>
 * mutation precision of the {@link org.apache.mahout.ga.watchmaker.cd.CDMutation Mutation} operator
 * <li>population size
 * <li>generations count<br>
 * number of generations the genetic algorithm will be run for.
 * 
 * </ul>
 */
public final class CDGA {

  private static final Logger log = LoggerFactory.getLogger(CDGA.class);

  private CDGA() {
  }

  public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(true).withShortName("i").withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create())
        .withDescription("The Path for input data directory.").create();

    Option labelOpt = obuilder.withLongName("label").withRequired(true).withShortName("l")
        .withArgument(abuilder.withName("index").withMinimum(1).withMaximum(1).create())
        .withDescription("label's index.").create();

    Option thresholdOpt = obuilder.withLongName("threshold").withRequired(false).withShortName("t").withArgument(
        abuilder.withName("threshold").withMinimum(1).withMaximum(1).create()).withDescription(
        "Condition activation threshold, default = 0.5.").create();

    Option crosspntsOpt = obuilder.withLongName("crosspnts").withRequired(false).withShortName("cp").withArgument(
        abuilder.withName("points").withMinimum(1).withMaximum(1).create()).withDescription(
        "Number of crossover points to use, default = 1.").create();

    Option mutrateOpt = obuilder.withLongName("mutrate").withRequired(true).withShortName("m").withArgument(
        abuilder.withName("true").withMinimum(1).withMaximum(1).create())
        .withDescription("Mutation rate (float).").create();

    Option mutrangeOpt = obuilder.withLongName("mutrange").withRequired(false).withShortName("mr").withArgument(
        abuilder.withName("range").withMinimum(1).withMaximum(1).create())
        .withDescription("Mutation range, default = 0.1 (10%).").create();

    Option mutprecOpt = obuilder.withLongName("mutprec").withRequired(false).withShortName("mp").withArgument(
        abuilder.withName("precision").withMinimum(1).withMaximum(1).create())
        .withDescription("Mutation precision, default = 2.").create();

    Option popsizeOpt = obuilder.withLongName("popsize").withRequired(true).withShortName("p").withArgument(
        abuilder.withName("size").withMinimum(1).withMaximum(1).create()).withDescription("Population size.").create();

    Option gencntOpt = obuilder.withLongName("gencnt").withRequired(true).withShortName("g").withArgument(
        abuilder.withName("count").withMinimum(1).withMaximum(1).create())
        .withDescription("Generations count.").create();

    Option helpOpt = DefaultOptionCreator.helpOption();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(helpOpt).withOption(labelOpt)
        .withOption(thresholdOpt).withOption(crosspntsOpt).withOption(mutrateOpt).withOption(mutrangeOpt)
        .withOption(mutprecOpt).withOption(popsizeOpt).withOption(gencntOpt).create();

    Parser parser = new Parser();
    parser.setGroup(group);

    try {
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }

      String dataset = cmdLine.getValue(inputOpt).toString();
      int target = Integer.parseInt(cmdLine.getValue(labelOpt).toString());
      double threshold =
          cmdLine.hasOption(thresholdOpt) ? Double.parseDouble(cmdLine.getValue(thresholdOpt).toString()) : 0.5;
      int crosspnts =
          cmdLine.hasOption(crosspntsOpt) ? Integer.parseInt(cmdLine.getValue(crosspntsOpt).toString()) : 1;
      double mutrate = Double.parseDouble(cmdLine.getValue(mutrateOpt).toString());
      double mutrange =
          cmdLine.hasOption(mutrangeOpt) ? Double.parseDouble(cmdLine.getValue(mutrangeOpt).toString()) : 0.1;
      int mutprec = cmdLine.hasOption(mutprecOpt) ? Integer.parseInt(cmdLine.getValue(mutprecOpt).toString()) : 2;
      int popSize = Integer.parseInt(cmdLine.getValue(popsizeOpt).toString());
      int genCount = Integer.parseInt(cmdLine.getValue(gencntOpt).toString());

      long start = System.currentTimeMillis();

      runJob(dataset, target, threshold, crosspnts, mutrate, mutrange, mutprec, popSize, genCount);

      long end = System.currentTimeMillis();

      printElapsedTime(end - start);
    } catch (OptionException e) {
      log.error("Error while parsing options", e);
      CommandLineUtil.printHelp(group);
    }
  }

  private static void runJob(String dataset,
                             int target,
                             double threshold,
                             int crosspnts,
                             double mutrate,
                             double mutrange,
                             int mutprec,
                             int popSize,
                             int genCount) throws IOException, InterruptedException, ClassNotFoundException {
    Path inpath = new Path(dataset);
    CDMahoutEvaluator.initializeDataSet(inpath);

    // Candidate Factory
    CandidateFactory<CDRule> factory = new CDFactory(threshold);

    // Evolution Scheme
    List<EvolutionaryOperator<CDRule>> operators = Lists.newArrayList();
    operators.add(new CDCrossover(crosspnts));
    operators.add(new CDMutation(mutrate, mutrange, mutprec));
    EvolutionPipeline<CDRule> pipeline = new EvolutionPipeline<CDRule>(operators);

    // 75 % of the dataset is dedicated to training
    DatasetSplit split = new DatasetSplit(0.75);

    // Fitness Evaluator (defaults to training)
    FitnessEvaluator<? super CDRule> evaluator = new CDFitnessEvaluator(dataset, target, split);
    // Selection Strategy
    SelectionStrategy<? super CDRule> selection = new RouletteWheelSelection();

    EvolutionEngine<CDRule> engine =
        new SequentialEvolutionEngine<CDRule>(factory, pipeline, evaluator, selection, RandomUtils.getRandom());

    engine.addEvolutionObserver(new EvolutionObserver<CDRule>() {
      @Override
      public void populationUpdate(PopulationData<? extends CDRule> data) {
        log.info("Generation {}", data.getGenerationNumber());
      }
    });

    // evolve the rules over the training set
    Rule solution = engine.evolve(popSize, 1, new GenerationCount(genCount));

    Path output = new Path("output");

    // fitness over the training set
    CDFitness bestTrainFit = CDMahoutEvaluator.evaluate(solution, target, inpath, output, split);

    // fitness over the testing set
    split.setTraining(false);
    CDFitness bestTestFit = CDMahoutEvaluator.evaluate(solution, target, inpath, output, split);

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
