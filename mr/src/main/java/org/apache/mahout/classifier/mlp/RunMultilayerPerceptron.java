/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.mlp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.csv.CSVUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

/** Run {@link MultilayerPerceptron} classification. */
public class RunMultilayerPerceptron {

  private static final Logger log = LoggerFactory.getLogger(RunMultilayerPerceptron.class);

  static class Parameters {
    String inputFilePathStr;
    String inputFileFormat;
    String modelFilePathStr;
    String outputFilePathStr;
    int columnStart;
    int columnEnd;
    boolean skipHeader;
  }
  
  public static void main(String[] args) throws Exception {
    
    Parameters parameters = new Parameters();
    
    if (parseArgs(args, parameters)) {
      log.info("Load model from {}.", parameters.modelFilePathStr);
      MultilayerPerceptron mlp = new MultilayerPerceptron(parameters.modelFilePathStr);

      log.info("Topology of MLP: {}.", Arrays.toString(mlp.getLayerSizeList().toArray()));

      // validate the data
      log.info("Read the data...");
      Path inputFilePath = new Path(parameters.inputFilePathStr);
      FileSystem inputFS = inputFilePath.getFileSystem(new Configuration());
      if (!inputFS.exists(inputFilePath)) {
        log.error("Input file '{}' does not exists!", parameters.inputFilePathStr);
        mlp.close();
        return;
      }

      Path outputFilePath = new Path(parameters.outputFilePathStr);
      FileSystem outputFS = inputFilePath.getFileSystem(new Configuration());
      if (outputFS.exists(outputFilePath)) {
        log.error("Output file '{}' already exists!", parameters.outputFilePathStr);
        mlp.close();
        return;
      }

      if (!parameters.inputFileFormat.equals("csv")) {
        log.error("Currently only supports for csv format.");
        mlp.close();
        return; // current only supports csv format
      }

      log.info("Read from column {} to column {}.", parameters.columnStart, parameters.columnEnd);

      BufferedWriter writer = null;
      BufferedReader reader = null;

      try {
        writer = new BufferedWriter(new OutputStreamWriter(outputFS.create(outputFilePath)));
        reader = new BufferedReader(new InputStreamReader(inputFS.open(inputFilePath)));
        
        String line;

        if (parameters.skipHeader) {
          reader.readLine();
        }

        while ((line = reader.readLine()) != null) {
          String[] tokens = CSVUtils.parseLine(line);
          double[] features = new double[Math.min(parameters.columnEnd, tokens.length) - parameters.columnStart + 1];

          for (int i = parameters.columnStart, j = 0; i < Math.min(parameters.columnEnd + 1, tokens.length); ++i, ++j) {
            features[j] = Double.parseDouble(tokens[i]);
          }
          Vector featureVec = new DenseVector(features);
          Vector res = mlp.getOutput(featureVec);
          int mostProbablyLabelIndex = res.maxValueIndex();
          writer.write(String.valueOf(mostProbablyLabelIndex));
        }
        mlp.close();
        log.info("Labeling finished.");
      } finally {
        Closeables.close(reader, true);
        Closeables.close(writer, true);
      }
    }
  }

  /**
   * Parse the arguments.
   *
   * @param args The input arguments.
   * @param parameters  The parameters need to be filled.
   * @return true or false
   * @throws Exception
   */
  private static boolean parseArgs(String[] args, Parameters parameters) throws Exception {
    // build the options
    log.info("Validate and parse arguments...");
    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    GroupBuilder groupBuilder = new GroupBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    Option inputFileFormatOption = optionBuilder
        .withLongName("format")
        .withShortName("f")
        .withArgument(argumentBuilder.withName("file type").withDefault("csv").withMinimum(1).withMaximum(1).create())
        .withDescription("type of input file, currently support 'csv'")
        .create();

    List<Integer> columnRangeDefault = Lists.newArrayList();
    columnRangeDefault.add(0);
    columnRangeDefault.add(Integer.MAX_VALUE);

    Option skipHeaderOption = optionBuilder.withLongName("skipHeader")
        .withShortName("sh").withRequired(false)
        .withDescription("whether to skip the first row of the input file")
        .create();

    Option inputColumnRangeOption = optionBuilder
        .withLongName("columnRange")
        .withShortName("cr")
        .withDescription("the column range of the input file, start from 0")
        .withArgument(argumentBuilder.withName("range").withMinimum(2).withMaximum(2)
            .withDefaults(columnRangeDefault).create()).create();

    Group inputFileTypeGroup = groupBuilder.withOption(skipHeaderOption)
        .withOption(inputColumnRangeOption).withOption(inputFileFormatOption)
        .create();

    Option inputOption = optionBuilder
        .withLongName("input")
        .withShortName("i")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("file path").withMinimum(1).withMaximum(1).create())
        .withDescription("the file path of unlabelled dataset")
        .withChildren(inputFileTypeGroup).create();

    Option modelOption = optionBuilder
        .withLongName("model")
        .withShortName("mo")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("model file").withMinimum(1).withMaximum(1).create())
        .withDescription("the file path of the model").create();

    Option labelsOption = optionBuilder
        .withLongName("labels")
        .withShortName("labels")
        .withArgument(argumentBuilder.withName("label-name").withMinimum(2).create())
        .withDescription("an ordered list of label names").create();

    Group labelsGroup = groupBuilder.withOption(labelsOption).create();

    Option outputOption = optionBuilder
        .withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withConsumeRemaining("file path").withMinimum(1).withMaximum(1).create())
        .withDescription("the file path of labelled results").withChildren(labelsGroup).create();

    // parse the input
    Parser parser = new Parser();
    Group normalOption = groupBuilder.withOption(inputOption).withOption(modelOption).withOption(outputOption).create();
    parser.setGroup(normalOption);
    CommandLine commandLine = parser.parseAndHelp(args);
    if (commandLine == null) {
      return false;
    }

    // obtain the arguments
    parameters.inputFilePathStr = TrainMultilayerPerceptron.getString(commandLine, inputOption);
    parameters.inputFileFormat = TrainMultilayerPerceptron.getString(commandLine, inputFileFormatOption);
    parameters.skipHeader = commandLine.hasOption(skipHeaderOption);
    parameters.modelFilePathStr = TrainMultilayerPerceptron.getString(commandLine, modelOption);
    parameters.outputFilePathStr = TrainMultilayerPerceptron.getString(commandLine, outputOption);

    List<?> columnRange = commandLine.getValues(inputColumnRangeOption);
    parameters.columnStart = Integer.parseInt(columnRange.get(0).toString());
    parameters.columnEnd = Integer.parseInt(columnRange.get(1).toString());

    return true;
  }

}
