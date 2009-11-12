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

package org.apache.mahout.df;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.builder.DefaultTreeBuilder;
import org.apache.mahout.df.callback.ForestPredictions;
import org.apache.mahout.df.callback.MeanTreeCollector;
import org.apache.mahout.df.callback.MultiCallback;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.DataLoader;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.ref.SequentialBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.uncommons.maths.Maths;

import java.io.IOException;
import java.util.Random;

/**
 * Test procedure as described in Breiman's paper.<br>
 * <b>Leo Breiman: Random Forests. Machine Learning 45(1): 5-32 (2001)</b>
 */
public class BreimanExample extends Configured implements Tool {

  private static final Logger log = LoggerFactory.getLogger(BreimanExample.class);

  /** sum test error */
  private static double sumTestErr;

  /** sum mean tree error */
  private static double sumTreeErr;

  /** sum test error with m=1 */
  private static double sumOneErr = 0.0;

  /** mean time to build a forest with m=log2(M)+1 */
  private static long sumTimeM;

  /** mean time to build a forest with m=1 */
  private static long sumTimeOne;

  /**
   * runs one iteration of the procedure.
   *
   * @param data training data
   * @param m number of random variables to select at each tree-node
   * @param nbtrees number of trees to grow
   * @throws Exception if an error occured while growing the trees
   */
  protected static void runIteration(Data data, int m, int nbtrees) {

    int dataSize = data.size();
    int nblabels = data.getDataset().nblabels();

    Random rng = RandomUtils.getRandom();

    Data train = data.clone();
    Data test = train.rsplit(rng, (int) (data.size() * 0.1));
    
    int[] trainLabels = train.extractLabels();
    int[] testLabels = test.extractLabels();
    
    DefaultTreeBuilder treeBuilder = new DefaultTreeBuilder();
    
    SequentialBuilder forestBuilder = new SequentialBuilder(rng, treeBuilder, train);

    // grow a forest with m = log2(M)+1
    ForestPredictions errorM = new ForestPredictions(dataSize, nblabels); // oob error when using m = log2(M)+1
    treeBuilder.setM(m);

    long time = System.currentTimeMillis();
    log.info("Growing a forest with m=" + m);
    DecisionForest forestM = forestBuilder.build(nbtrees, errorM);
    sumTimeM += System.currentTimeMillis() - time;

    double oobM = ErrorEstimate.errorRate(trainLabels, errorM.computePredictions(rng)); // oob error estimate when m = log2(M)+1

    // grow a forest with m=1
    ForestPredictions errorOne = new ForestPredictions(dataSize, nblabels); // oob error when using m = 1
    treeBuilder.setM(1);

    time = System.currentTimeMillis();
    log.info("Growing a forest with m=1");
    DecisionForest forestOne = forestBuilder.build(nbtrees, errorOne);
    sumTimeOne += System.currentTimeMillis() - time;

    double oobOne = ErrorEstimate.errorRate(trainLabels, errorOne.computePredictions(rng)); // oob error estimate when m = 1

    // compute the test set error (Selection Error), and mean tree error (One Tree Error),
    // using the lowest oob error forest
    ForestPredictions testError = new ForestPredictions(dataSize, nblabels); // test set error
    MeanTreeCollector treeError = new MeanTreeCollector(train, nbtrees); // mean tree error

    // compute the test set error using m=1 (Single Input Error)
    errorOne = new ForestPredictions(dataSize, nblabels);

    if (oobM < oobOne) {
      forestM.classify(test, new MultiCallback(testError, treeError));
      forestOne.classify(test, errorOne);
    } else {
      forestOne.classify(test,
          new MultiCallback(testError, treeError, errorOne));
    }

    sumTestErr += ErrorEstimate.errorRate(testLabels, testError.computePredictions(rng));
    sumOneErr += ErrorEstimate.errorRate(testLabels, errorOne.computePredictions(rng));
    sumTreeErr += treeError.meanTreeError();
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new BreimanExample(), args);
  }

  @Override
  public int run(String[] args) throws IOException {
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true)
        .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
        .withDescription("Data path").create();

    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true)
        .withArgument(abuilder.withName("dataset").withMinimum(1).withMaximum(1).create())
        .withDescription("Dataset path").create();

    Option nbtreesOpt = obuilder.withLongName("nbtrees").withShortName("t").withRequired(true)
        .withArgument(abuilder.withName("nbtrees").withMinimum(1).withMaximum(1).create())
        .withDescription("Number of trees to grow, each iteration").create();

    Option nbItersOpt = obuilder.withLongName("iterations").withShortName("i").withRequired(true)
        .withArgument(abuilder.withName("numIterations").withMinimum(1).withMaximum(1).create())
        .withDescription("Number of times to repeat the test").create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help")
        .withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(dataOpt).withOption(datasetOpt)
        .withOption(nbItersOpt).withOption(nbtreesOpt).withOption(helpOpt).create();

    Path dataPath;
    Path datasetPath;
    int nbTrees;
    int nbIterations;
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption("help")) {
        CommandLineUtil.printHelp(group);
        return -1;
      }

      String dataName = cmdLine.getValue(dataOpt).toString();
      String datasetName = cmdLine.getValue(datasetOpt).toString();
      nbTrees = Integer.parseInt(cmdLine.getValue(nbtreesOpt).toString());
      nbIterations = Integer.parseInt(cmdLine.getValue(nbItersOpt).toString());

      dataPath = new Path(dataName);
      datasetPath = new Path(datasetName);
    } catch (OptionException e) {
      System.err.println("Exception : " + e);
      CommandLineUtil.printHelp(group);
      return -1;
    }
    
    // load the data
    FileSystem fs = dataPath.getFileSystem(new Configuration());
    Dataset dataset = Dataset.load(getConf(), datasetPath);
    Data data = DataLoader.loadData(dataset, fs, dataPath);

    // take m to be the first integer less than log2(M) + 1, where M is the
    // number of inputs
    int m = (int) Math.floor(Maths.log(2, data.getDataset().nbAttributes()) + 1);

    for (int iteration = 0; iteration < nbIterations; iteration++) {
      log.info("Iteration " + iteration);
      runIteration(data, m, nbTrees);
    }

    log.info("********************************************");
    log.info("Selection error : " + sumTestErr / nbIterations);
    log.info("Single Input error : " + sumOneErr / nbIterations);
    log.info("One Tree error : " + sumTreeErr / nbIterations);
    log.info("");
    log.info("Mean Random Input Time : " + DFUtils.elapsedTime(sumTimeM / nbIterations));
    log.info("Mean Single Input Time : " + DFUtils.elapsedTime(sumTimeOne / nbIterations));

    return 0;
  }

}
