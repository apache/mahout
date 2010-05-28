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

package org.apache.mahout.df.mapreduce;

import java.io.IOException;
import java.util.Random;

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
import org.apache.mahout.df.DFUtils;
import org.apache.mahout.df.DecisionForest;
import org.apache.mahout.df.ErrorEstimate;
import org.apache.mahout.df.builder.DefaultTreeBuilder;
import org.apache.mahout.df.callback.ForestPredictions;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.DataLoader;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.mapreduce.inmem.InMemBuilder;
import org.apache.mahout.df.mapreduce.partial.PartialBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tool to builds a Random Forest using any given dataset (in UCI format). Can use either the in-mem mapred or
 * partial mapred implementations. Stores the forest in the given output directory
 */
public class BuildForest extends Configured implements Tool {
  
  private static final Logger log = LoggerFactory.getLogger(BuildForest.class);
  
  private Path dataPath;
  
  private Path datasetPath;

  private Path outputPath;
  private int m; // Number of variables to select at each tree-node
  
  private int nbTrees; // Number of trees to grow
  
  private Long seed; // Random seed
  
  private boolean isPartial; // use partial data implementation
  
  private boolean isOob; // estimate oob error;

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option oobOpt = obuilder.withShortName("oob").withRequired(false).withDescription(
      "Optional, estimate the out-of-bag error").create();
    
    Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true).withArgument(
      abuilder.withName("path").withMinimum(1).withMaximum(1).create()).withDescription("Data path").create();
    
    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true).withArgument(
      abuilder.withName("dataset").withMinimum(1).withMaximum(1).create()).withDescription("Dataset path")
        .create();
    
    Option selectionOpt = obuilder.withLongName("selection").withShortName("sl").withRequired(true)
        .withArgument(abuilder.withName("m").withMinimum(1).withMaximum(1).create()).withDescription(
          "Number of variables to select randomly at each tree-node").create();
    
    Option seedOpt = obuilder.withLongName("seed").withShortName("sd").withRequired(false).withArgument(
      abuilder.withName("seed").withMinimum(1).withMaximum(1).create()).withDescription(
      "Optional, seed value used to initialise the Random number generator").create();
    
    Option partialOpt = obuilder.withLongName("partial").withShortName("p").withRequired(false)
        .withDescription("Optional, use the Partial Data implementation").create();
    
    Option nbtreesOpt = obuilder.withLongName("nbtrees").withShortName("t").withRequired(true).withArgument(
      abuilder.withName("nbtrees").withMinimum(1).withMaximum(1).create()).withDescription(
      "Number of trees to grow").create();
    
    Option outputOpt = obuilder.withLongName("output").withShortName("o").withRequired(true).withArgument(
        abuilder.withName("path").withMinimum(1).withMaximum(1).create()).
        withDescription("Output path, will contain the Decision Forest").create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();
    
    Group group = gbuilder.withName("Options").withOption(oobOpt).withOption(dataOpt).withOption(datasetOpt)
        .withOption(selectionOpt).withOption(seedOpt).withOption(partialOpt).withOption(nbtreesOpt)
        .withOption(outputOpt).withOption(helpOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption("help")) {
        CommandLineUtil.printHelp(group);
        return -1;
      }
      
      isPartial = cmdLine.hasOption(partialOpt);
      isOob = cmdLine.hasOption(oobOpt);
      String dataName = cmdLine.getValue(dataOpt).toString();
      String datasetName = cmdLine.getValue(datasetOpt).toString();
      String outputName = cmdLine.getValue(outputOpt).toString();
      m = Integer.parseInt(cmdLine.getValue(selectionOpt).toString());
      nbTrees = Integer.parseInt(cmdLine.getValue(nbtreesOpt).toString());
      
      if (cmdLine.hasOption(seedOpt)) {
        seed = Long.valueOf(cmdLine.getValue(seedOpt).toString());
      }
      
      log.debug("data : {}", dataName);
      log.debug("dataset : {}", datasetName);
      log.debug("output : {}", outputName);
      log.debug("m : {}", m);
      log.debug("seed : {}", seed);
      log.debug("nbtrees : {}", nbTrees);
      log.debug("isPartial : {}", isPartial);
      log.debug("isOob : {}", isOob);
      
      dataPath = new Path(dataName);
      datasetPath = new Path(datasetName);
      outputPath = new Path(outputName);
      
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
      return -1;
    }
    
    buildForest();
    
    return 0;
  }
  
  private void buildForest() throws IOException, ClassNotFoundException, InterruptedException {
    // make sure the output path does not exist
    FileSystem ofs = outputPath.getFileSystem(getConf());
    if (ofs.exists(outputPath)) {
      log.error("Output path already exists");
      return;
    }

    DefaultTreeBuilder treeBuilder = new DefaultTreeBuilder();
    treeBuilder.setM(m);
    
    Dataset dataset = Dataset.load(getConf(), datasetPath);
    
    ForestPredictions callback = isOob ? new ForestPredictions(dataset.nbInstances(), dataset.nblabels())
        : null;
    
    Builder forestBuilder;
    
    if (isPartial) {
      log.info("Partial Mapred implementation");
      forestBuilder = new PartialBuilder(treeBuilder, dataPath, datasetPath, seed, getConf());
    } else {
      log.info("InMem Mapred implementation");
      forestBuilder = new InMemBuilder(treeBuilder, dataPath, datasetPath, seed, getConf());
    }

    forestBuilder.setOutputDirName(outputPath.getName());
    
    log.info("Building the forest...");
    long time = System.currentTimeMillis();
    
    DecisionForest forest = forestBuilder.build(nbTrees, callback);
    
    time = System.currentTimeMillis() - time;
    log.info("Build Time: {}", DFUtils.elapsedTime(time));
    
    if (isOob) {
      Random rng;
      if (seed != null) {
        rng = RandomUtils.getRandom(seed);
      } else {
        rng = RandomUtils.getRandom();
      }
      
      FileSystem fs = dataPath.getFileSystem(getConf());
      int[] labels = Data.extractLabels(dataset, fs, dataPath);
      
      log.info("oob error estimate : "
                           + ErrorEstimate.errorRate(labels, callback.computePredictions(rng)));
    }

    // store the decision forest in the output path
    Path forestPath = new Path(outputPath, "forest.seq");
    log.info("Storing the forest in: " + forestPath);
    DFUtils.storeWritable(getConf(), forestPath, forest);

  }
  
  protected static Data loadData(Configuration conf, Path dataPath, Dataset dataset) throws IOException {
    log.info("Loading the data...");
    FileSystem fs = dataPath.getFileSystem(conf);
    Data data = DataLoader.loadData(dataset, fs, dataPath);
    log.info("Data Loaded");
    
    return data;
  }
  
  /**
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new BuildForest(), args);
  }
  
}
