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

package org.apache.mahout.classifier.df;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.math3.util.FastMath;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.classifier.df.builder.DefaultTreeBuilder;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.ref.SequentialBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Test procedure as described in Breiman's paper.<br>
 * <b>Leo Breiman: Random Forests. Machine Learning 45(1): 5-32 (2001)</b>
 */
public class BreimanExample extends Configured implements Tool {
  
  private static final Logger log = LoggerFactory.getLogger(BreimanExample.class);
  
  /** sum test error */
  private double sumTestErrM;

  private double sumTestErrOne;
  
  /** mean time to build a forest with m=log2(M)+1 */
  private long sumTimeM;
  
  /** mean time to build a forest with m=1 */
  private long sumTimeOne;
  
  /** mean number of nodes for all the trees grown with m=log2(M)+1 */
  private long numNodesM;
  
  /** mean number of nodes for all the trees grown with m=1 */
  private long numNodesOne;
  
  /**
   * runs one iteration of the procedure.
   * 
   * @param rng
   *          random numbers generator
   * @param data
   *          training data
   * @param m
   *          number of random variables to select at each tree-node
   * @param nbtrees
   *          number of trees to grow
   */
  private void runIteration(Random rng, Data data, int m, int nbtrees) {
    
    log.info("Splitting the data");
    Data train = data.clone();
    Data test = train.rsplit(rng, (int) (data.size() * 0.1));
    
    DefaultTreeBuilder treeBuilder = new DefaultTreeBuilder();
    
    SequentialBuilder forestBuilder = new SequentialBuilder(rng, treeBuilder, train);
    
    // grow a forest with m = log2(M)+1
    treeBuilder.setM(m);
    
    long time = System.currentTimeMillis();
    log.info("Growing a forest with m={}", m);
    DecisionForest forestM = forestBuilder.build(nbtrees);
    sumTimeM += System.currentTimeMillis() - time;
    numNodesM += forestM.nbNodes();
    
    // grow a forest with m=1
    treeBuilder.setM(1);
    
    time = System.currentTimeMillis();
    log.info("Growing a forest with m=1");
    DecisionForest forestOne = forestBuilder.build(nbtrees);
    sumTimeOne += System.currentTimeMillis() - time;
    numNodesOne += forestOne.nbNodes();
    
    // compute the test set error (Selection Error), and mean tree error (One Tree Error),
    double[] testLabels = test.extractLabels();
    double[][] predictions = new double[test.size()][];
    
    forestM.classify(test, predictions);
    double[] sumPredictions = new double[test.size()];
    Arrays.fill(sumPredictions, 0.0);
    for (int i = 0; i < predictions.length; i++) {
      for (int j = 0; j < predictions[i].length; j++) {
        sumPredictions[i] += predictions[i][j];
      }
    }
    sumTestErrM += ErrorEstimate.errorRate(testLabels, sumPredictions);
    
    forestOne.classify(test, predictions);
    Arrays.fill(sumPredictions, 0.0);
    for (int i = 0; i < predictions.length; i++) {
      for (int j = 0; j < predictions[i].length; j++) {
        sumPredictions[i] += predictions[i][j];
      }
    }
    sumTestErrOne += ErrorEstimate.errorRate(testLabels, sumPredictions);
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new BreimanExample(), args);
  }
  
  @Override
  public int run(String[] args) throws IOException {
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true).withArgument(
      abuilder.withName("path").withMinimum(1).withMaximum(1).create()).withDescription("Data path").create();
    
    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true).withArgument(
      abuilder.withName("dataset").withMinimum(1).withMaximum(1).create()).withDescription("Dataset path")
        .create();
    
    Option nbtreesOpt = obuilder.withLongName("nbtrees").withShortName("t").withRequired(true).withArgument(
      abuilder.withName("nbtrees").withMinimum(1).withMaximum(1).create()).withDescription(
      "Number of trees to grow, each iteration").create();
    
    Option nbItersOpt = obuilder.withLongName("iterations").withShortName("i").withRequired(true)
        .withArgument(abuilder.withName("numIterations").withMinimum(1).withMaximum(1).create())
        .withDescription("Number of times to repeat the test").create();
    
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();
    
    Group group = gbuilder.withName("Options").withOption(dataOpt).withOption(datasetOpt).withOption(
      nbItersOpt).withOption(nbtreesOpt).withOption(helpOpt).create();
    
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
      log.error("Error while parsing options", e);
      CommandLineUtil.printHelp(group);
      return -1;
    }
    
    // load the data
    FileSystem fs = dataPath.getFileSystem(new Configuration());
    Dataset dataset = Dataset.load(getConf(), datasetPath);
    Data data = DataLoader.loadData(dataset, fs, dataPath);
    
    // take m to be the first integer less than log2(M) + 1, where M is the
    // number of inputs
    int m = (int) Math.floor(FastMath.log(2.0, data.getDataset().nbAttributes()) + 1);
    
    Random rng = RandomUtils.getRandom();
    for (int iteration = 0; iteration < nbIterations; iteration++) {
      log.info("Iteration {}", iteration);
      runIteration(rng, data, m, nbTrees);
    }
    
    log.info("********************************************");
    log.info("Random Input Test Error : {}", sumTestErrM / nbIterations);
    log.info("Single Input Test Error : {}", sumTestErrOne / nbIterations);
    log.info("Mean Random Input Time : {}", DFUtils.elapsedTime(sumTimeM / nbIterations));
    log.info("Mean Single Input Time : {}", DFUtils.elapsedTime(sumTimeOne / nbIterations));
    log.info("Mean Random Input Num Nodes : {}", numNodesM / nbIterations);
    log.info("Mean Single Input Num Nodes : {}", numNodesOne / nbIterations);
    
    return 0;
  }
  
}
