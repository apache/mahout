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

package org.apache.mahout.df.mapreduce;

import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import java.util.Arrays;

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
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.DFUtils;
import org.apache.mahout.df.DecisionForest;
import org.apache.mahout.df.ErrorEstimate;
import org.apache.mahout.df.builder.DefaultTreeBuilder;
import org.apache.mahout.df.callback.ForestPredictions;
import org.apache.mahout.df.data.*;
import org.apache.mahout.df.mapreduce.inmem.InMemBuilder;
import org.apache.mahout.df.mapreduce.partial.PartialBuilder;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.ClassifierResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tool to classify a Dataset using a previously built Decision Forest
 */
public class TestForest extends Configured implements Tool {

  private static final Logger log = LoggerFactory.getLogger(TestForest.class);

  private Path dataPath; // test data path

  private Path datasetPath;

  private Path modelPath; // path where the forest is stored

  private Path outputPath; // path to predictions file, if null do not output the predictions

  private boolean analyze; // analyze the classification results ?

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withShortName("i").withRequired(true).withArgument(
      abuilder.withName("path").withMinimum(1).withMaximum(1).create()).withDescription("Test data path").create();

    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true).withArgument(
      abuilder.withName("dataset").withMinimum(1).withMaximum(1).create()).withDescription("Dataset path")
        .create();

    Option modelOpt = obuilder.withLongName("model").withShortName("m").withRequired(true).withArgument(
        abuilder.withName("path").withMinimum(1).withMaximum(1).create()).
        withDescription("Path to the Decision Forest").create();

    Option outputOpt = obuilder.withLongName("output").withShortName("o").withRequired(false).withArgument(
      abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
      "Path to generated predictions file").create();

    Option analyzeOpt = obuilder.withLongName("analyze").withShortName("a").withRequired(false).create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(datasetOpt)
        .withOption(modelOpt).withOption(outputOpt).withOption(analyzeOpt).withOption(helpOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption("help")) {
        CommandLineUtil.printHelp(group);
        return -1;
      }

      String dataName = cmdLine.getValue(inputOpt).toString();
      String datasetName = cmdLine.getValue(datasetOpt).toString();
      String modelName = cmdLine.getValue(modelOpt).toString();
      String outputName = (cmdLine.hasOption(outputOpt)) ? cmdLine.getValue(outputOpt).toString() : null;
      analyze = cmdLine.hasOption(analyzeOpt);

      log.debug("inout   : {}", dataName);
      log.debug("dataset : {}", datasetName);
      log.debug("model   : {}", modelName);
      log.debug("output  : {}", outputName);
      log.debug("analyze : {}", analyze);

      dataPath = new Path(dataName);
      datasetPath = new Path(datasetName);
      modelPath = new Path(modelName);
      if (outputName != null) {
        outputPath = new Path(outputName);
      }
    } catch (OptionException e) {
      System.err.println("Exception : " + e);
      CommandLineUtil.printHelp(group);
      return -1;
    }

    testForest();

    return 0;
  }

  private void testForest() throws IOException, ClassNotFoundException, InterruptedException {

    FileSystem ofs = null;

    // make sure the output file does not exist
    if (outputPath != null) {
      ofs = outputPath.getFileSystem(getConf());
      if (ofs.exists(outputPath)) {
        throw new IllegalArgumentException("Output path already exists");
      }
    }

    Dataset dataset = Dataset.load(getConf(), datasetPath);
    DataConverter converter = new DataConverter(dataset);

    log.info("Loading the forest...");
    FileSystem fs = modelPath.getFileSystem(getConf());
    Path[] modelfiles = DFUtils.listOutputFiles(fs, modelPath);
    DecisionForest forest = null;
    for (Path path : modelfiles) {
      FSDataInputStream dataInput = new FSDataInputStream(fs.open(path));
      if (forest == null) {
        forest = DecisionForest.read(dataInput);
      } else {
        forest.readFields(dataInput);
      }

      dataInput.close();
    }    

    if (forest == null) {
      log.error("No Decision Forest found!");
      return;
    }

    // create the predictions file
    FSDataOutputStream ofile = (outputPath != null) ? ofs.create(outputPath) : null;

    log.info("Sequential classification...");
    long time = System.currentTimeMillis();

    FileSystem tfs = dataPath.getFileSystem(getConf());
    FSDataInputStream input = tfs.open(dataPath);
    Scanner scanner = new Scanner(input);
    Random rng = RandomUtils.getRandom();
    ResultAnalyzer analyzer = (analyze) ? new ResultAnalyzer(Arrays.asList(dataset.labels()), "unknown") : null;

    while (scanner.hasNextLine()) {
      String line = scanner.nextLine();
      if (line.isEmpty()) {
        continue; // skip empty lines
      }

      Instance instance = converter.convert(0, line);
      int prediction = forest.classify(rng, instance);

      if (outputPath != null) {
        ofile.writeChars(Integer.toString(prediction)); // write the prediction
        ofile.writeChar('\n');
      }
      
      if (analyze) {
        analyzer.addInstance(dataset.getLabel(instance.label), new ClassifierResult(dataset.getLabel(prediction), 1.0));
      }
    }

    time = System.currentTimeMillis() - time;
    log.info("Classification Time: {}", DFUtils.elapsedTime(time));

    if (analyze) {
      log.info(analyzer.summarize());
    }
  }

  /**
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new TestForest(), args);
  }

}