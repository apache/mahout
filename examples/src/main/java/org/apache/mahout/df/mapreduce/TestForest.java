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
import java.util.Scanner;
import java.util.Arrays;

import com.google.common.io.Closeables;
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
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.df.data.DataConverter;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.data.Instance;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tool to classify a Dataset using a previously built Decision Forest
 */
public class TestForest extends Configured implements Tool {

  private static final Logger log = LoggerFactory.getLogger(TestForest.class);

  private FileSystem dataFS;
  private Path dataPath; // test data path

  private Path datasetPath;

  private Path modelPath; // path where the forest is stored

  private FileSystem outFS;
  private Path outputPath; // path to predictions file, if null do not output the predictions

  private boolean analyze; // analyze the classification results ?

  private boolean useMapreduce; // use the mapreduce classifier ?

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

    Option mrOpt = obuilder.withLongName("mapreduce").withShortName("mr").withRequired(false).create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(datasetOpt).withOption(modelOpt)
        .withOption(outputOpt).withOption(analyzeOpt).withOption(mrOpt).withOption(helpOpt).create();

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
      String outputName = cmdLine.hasOption(outputOpt) ? cmdLine.getValue(outputOpt).toString() : null;
      analyze = cmdLine.hasOption(analyzeOpt);
      useMapreduce = cmdLine.hasOption(mrOpt);

      log.debug("inout     : {}", dataName);
      log.debug("dataset   : {}", datasetName);
      log.debug("model     : {}", modelName);
      log.debug("output    : {}", outputName);
      log.debug("analyze   : {}", analyze);
      log.debug("mapreduce : {}", useMapreduce);

      dataPath = new Path(dataName);
      datasetPath = new Path(datasetName);
      modelPath = new Path(modelName);
      if (outputName != null) {
        outputPath = new Path(outputName);
      }
    } catch (OptionException e) {
      log.warn(e.toString(), e);
      CommandLineUtil.printHelp(group);
      return -1;
    }

    testForest();

    return 0;
  }

  private void testForest() throws IOException, ClassNotFoundException, InterruptedException {

    // make sure the output file does not exist
    if (outputPath != null) {
      outFS = outputPath.getFileSystem(getConf());
      if (outFS.exists(outputPath)) {
        throw new IllegalArgumentException("Output path already exists");
      }
    }

    // make sure the decision forest exists
    FileSystem mfs = modelPath.getFileSystem(getConf());
    if (!mfs.exists(modelPath)) {
      throw new IllegalArgumentException("The forest path does not exist");
    }

    // make sure the test data exists
    dataFS = dataPath.getFileSystem(getConf());
    if (!dataFS.exists(dataPath)) {
      throw new IllegalArgumentException("The Test data path does not exist");
    }

    if (useMapreduce) {
      mapreduce();
    } else {
      sequential();
    }

  }

  private void mapreduce() throws ClassNotFoundException, IOException, InterruptedException {
    if (outputPath == null) {
      throw new IllegalArgumentException("You must specify the ouputPath when using the mapreduce implementation");
    }

    Classifier classifier = new Classifier(modelPath, dataPath, datasetPath, outputPath, getConf(), analyze);

    classifier.run();

    if (analyze) {
      log.info("{}", classifier.getAnalyzer());
    }
  }

  private void sequential() throws IOException {

    log.info("Loading the forest...");
    DecisionForest forest = DecisionForest.load(getConf(), modelPath);

    if (forest == null) {
      log.error("No Decision Forest found!");
      return;
    }

    // load the dataset
    Dataset dataset = Dataset.load(getConf(), datasetPath);
    DataConverter converter = new DataConverter(dataset);

    log.info("Sequential classification...");
    long time = System.currentTimeMillis();

    Random rng = RandomUtils.getRandom();
    ResultAnalyzer analyzer = analyze ? new ResultAnalyzer(Arrays.asList(dataset.labels()), "unknown") : null;

    if (dataFS.getFileStatus(dataPath).isDir()) {
      //the input is a directory of files
      testDirectory(outputPath, converter, forest, dataset, analyzer, rng);
    }  else {
      // the input is one single file
      testFile(dataPath, outputPath, converter, forest, dataset, analyzer, rng);
    }

    time = System.currentTimeMillis() - time;
    log.info("Classification Time: {}", DFUtils.elapsedTime(time));

    if (analyzer != null) {
      log.info("{}", analyzer);
    }
  }

  private void testDirectory(Path outPath, DataConverter converter, DecisionForest forest, Dataset dataset,
                        ResultAnalyzer analyzer, Random rng) throws IOException {
    Path[] infiles = DFUtils.listOutputFiles(dataFS, dataPath);

    for (Path path : infiles) {
      log.info("Classifying : " + path);
      Path outfile = outPath != null ? new Path(outPath, path.getName()).suffix(".out") : null;
      testFile(path, outfile, converter, forest, dataset, analyzer, rng);
    }
  }

  private void testFile(Path inPath, Path outPath, DataConverter converter, DecisionForest forest, Dataset dataset,
                        ResultAnalyzer analyzer, Random rng) throws IOException {
    // create the predictions file
    FSDataOutputStream ofile = null;

    if (outPath != null) {
      ofile = outFS.create(outPath);
    }

    FSDataInputStream input = dataFS.open(inPath);
    try {
      Scanner scanner = new Scanner(input);

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

        if (analyzer != null) {
          analyzer.addInstance(dataset.getLabel(instance.getLabel()),
                               new ClassifierResult(dataset.getLabel(prediction), 1.0));
        }
      }

      scanner.close();
    } finally {
      Closeables.closeQuietly(input);
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new TestForest(), args);
  }

}