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
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.Arrays;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;

/** Train a {@link MultilayerPerceptron}. */
public final class TrainMultilayerPerceptron {

  private static final Logger log = LoggerFactory.getLogger(TrainMultilayerPerceptron.class);
  
  /**  The parameters used by MLP. */
  static class Parameters {
    double learningRate;
    double momemtumWeight;
    double regularizationWeight;

    String inputFilePath;
    boolean skipHeader;
    Map<String, Integer> labelsIndex = Maps.newHashMap();

    String modelFilePath;
    boolean updateModel;
    List<Integer> layerSizeList = Lists.newArrayList();
    String squashingFunctionName;
  }

  /*
  private double learningRate;
  private double momemtumWeight;
  private double regularizationWeight;

  private String inputFilePath;
  private boolean skipHeader;
  private Map<String, Integer> labelsIndex = Maps.newHashMap();

  private String modelFilePath;
  private boolean updateModel;
  private List<Integer> layerSizeList = Lists.newArrayList();
  private String squashingFunctionName;*/

  public static void main(String[] args) throws Exception {
    Parameters parameters = new Parameters();
    
    if (parseArgs(args, parameters)) {
      log.info("Validate model...");
      // check whether the model already exists
      Path modelPath = new Path(parameters.modelFilePath);
      FileSystem modelFs = modelPath.getFileSystem(new Configuration());
      MultilayerPerceptron mlp;

      if (modelFs.exists(modelPath) && parameters.updateModel) {
        // incrementally update existing model
        log.info("Build model from existing model...");
        mlp = new MultilayerPerceptron(parameters.modelFilePath);
      } else {
        if (modelFs.exists(modelPath)) {
          modelFs.delete(modelPath, true); // delete the existing file
        }
        log.info("Build model from scratch...");
        mlp = new MultilayerPerceptron();
        for (int i = 0; i < parameters.layerSizeList.size(); ++i) {
          if (i != parameters.layerSizeList.size() - 1) {
            mlp.addLayer(parameters.layerSizeList.get(i), false, parameters.squashingFunctionName);
          } else {
            mlp.addLayer(parameters.layerSizeList.get(i), true, parameters.squashingFunctionName);
          }
          mlp.setCostFunction("Minus_Squared");
          mlp.setLearningRate(parameters.learningRate)
             .setMomentumWeight(parameters.momemtumWeight)
             .setRegularizationWeight(parameters.regularizationWeight);
        }
        mlp.setModelPath(parameters.modelFilePath);
      }

      // set the parameters
      mlp.setLearningRate(parameters.learningRate)
         .setMomentumWeight(parameters.momemtumWeight)
         .setRegularizationWeight(parameters.regularizationWeight);

      // train by the training data
      Path trainingDataPath = new Path(parameters.inputFilePath);
      FileSystem dataFs = trainingDataPath.getFileSystem(new Configuration());

      Preconditions.checkArgument(dataFs.exists(trainingDataPath), "Training dataset %s cannot be found!",
                                  parameters.inputFilePath);

      log.info("Read data and train model...");
      BufferedReader reader = null;

      try {
        reader = new BufferedReader(new InputStreamReader(dataFs.open(trainingDataPath)));
        String line;

        // read training data line by line
        if (parameters.skipHeader) {
          reader.readLine();
        }

        int labelDimension = parameters.labelsIndex.size();
        while ((line = reader.readLine()) != null) {
          String[] token = line.split(",");
          String label = token[token.length - 1];
          int labelIndex = parameters.labelsIndex.get(label);

          double[] instances = new double[token.length - 1 + labelDimension];
          for (int i = 0; i < token.length - 1; ++i) {
            instances[i] = Double.parseDouble(token[i]);
          }
          for (int i = 0; i < labelDimension; ++i) {
            instances[token.length - 1 + i] = 0;
          }
          // set the corresponding dimension
          instances[token.length - 1 + labelIndex] = 1;

          Vector instance = new DenseVector(instances).viewPart(0, instances.length);
          mlp.trainOnline(instance);
        }

        // write model back
        log.info("Write trained model to {}", parameters.modelFilePath);
        mlp.writeModelToFile();
        mlp.close();
      } finally {
        Closeables.close(reader, true);
      }
    }
  }

  /**
   * Parse the input arguments.
   * 
   * @param args The input arguments
   * @param parameters The parameters parsed.
   * @return Whether the input arguments are valid.
   * @throws Exception
   */
  private static boolean parseArgs(String[] args, Parameters parameters) throws Exception {
    // build the options
    log.info("Validate and parse arguments...");
    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    GroupBuilder groupBuilder = new GroupBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    // whether skip the first row of the input file
    Option skipHeaderOption = optionBuilder.withLongName("skipHeader")
        .withShortName("sh").create();

    Group skipHeaderGroup = groupBuilder.withOption(skipHeaderOption).create();

    Option inputOption = optionBuilder
        .withLongName("input")
        .withShortName("i")
        .withRequired(true)
        .withChildren(skipHeaderGroup)
        .withArgument(argumentBuilder.withName("path").withMinimum(1).withMaximum(1)
                .create()).withDescription("the file path of training dataset")
        .create();

    Option labelsOption = optionBuilder
        .withLongName("labels")
        .withShortName("labels")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("label-name").withMinimum(2).create())
        .withDescription("label names").create();

    Option updateOption = optionBuilder
        .withLongName("update")
        .withShortName("u")
        .withDescription("whether to incrementally update model if the model exists")
        .create();

    Group modelUpdateGroup = groupBuilder.withOption(updateOption).create();

    Option modelOption = optionBuilder
        .withLongName("model")
        .withShortName("mo")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("model-path").withMinimum(1).withMaximum(1).create())
        .withDescription("the path to store the trained model")
        .withChildren(modelUpdateGroup).create();

    Option layerSizeOption = optionBuilder
        .withLongName("layerSize")
        .withShortName("ls")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("size of layer").withMinimum(2).withMaximum(5).create())
        .withDescription("the size of each layer").create();

    Option squashingFunctionOption = optionBuilder
        .withLongName("squashingFunction")
        .withShortName("sf")
        .withArgument(argumentBuilder.withName("squashing function").withMinimum(1).withMaximum(1)
            .withDefault("Sigmoid").create())
        .withDescription("the name of squashing function (currently only supports Sigmoid)")
        .create();

    Option learningRateOption = optionBuilder
        .withLongName("learningRate")
        .withShortName("l")
        .withArgument(argumentBuilder.withName("learning rate").withMaximum(1)
            .withMinimum(1).withDefault(NeuralNetwork.DEFAULT_LEARNING_RATE).create())
        .withDescription("learning rate").create();

    Option momemtumOption = optionBuilder
        .withLongName("momemtumWeight")
        .withShortName("m")
        .withArgument(argumentBuilder.withName("momemtum weight").withMaximum(1)
            .withMinimum(1).withDefault(NeuralNetwork.DEFAULT_MOMENTUM_WEIGHT).create())
        .withDescription("momemtum weight").create();

    Option regularizationOption = optionBuilder
        .withLongName("regularizationWeight")
        .withShortName("r")
        .withArgument(argumentBuilder.withName("regularization weight").withMaximum(1)
            .withMinimum(1).withDefault(NeuralNetwork.DEFAULT_REGULARIZATION_WEIGHT).create())
        .withDescription("regularization weight").create();

    // parse the input
    Parser parser = new Parser();
    Group normalOptions = groupBuilder.withOption(inputOption)
        .withOption(skipHeaderOption).withOption(updateOption)
        .withOption(labelsOption).withOption(modelOption)
        .withOption(layerSizeOption).withOption(squashingFunctionOption)
        .withOption(learningRateOption).withOption(momemtumOption)
        .withOption(regularizationOption).create();

    parser.setGroup(normalOptions);

    CommandLine commandLine = parser.parseAndHelp(args);
    if (commandLine == null) {
      return false;
    }

    parameters.learningRate = getDouble(commandLine, learningRateOption);
    parameters.momemtumWeight = getDouble(commandLine, momemtumOption);
    parameters.regularizationWeight = getDouble(commandLine, regularizationOption);

    parameters.inputFilePath = getString(commandLine, inputOption);
    parameters.skipHeader = commandLine.hasOption(skipHeaderOption);

    List<String> labelsList = getStringList(commandLine, labelsOption);
    int currentIndex = 0;
    for (String label : labelsList) {
      parameters.labelsIndex.put(label, currentIndex++);
    }

    parameters.modelFilePath = getString(commandLine, modelOption);
    parameters.updateModel = commandLine.hasOption(updateOption);

    parameters.layerSizeList = getIntegerList(commandLine, layerSizeOption);

    parameters.squashingFunctionName = getString(commandLine, squashingFunctionOption);

    System.out.printf("Input: %s, Model: %s, Update: %s, Layer size: %s, Squashing function: %s, Learning rate: %f," +
        " Momemtum weight: %f, Regularization Weight: %f\n", parameters.inputFilePath, parameters.modelFilePath, 
        parameters.updateModel, Arrays.toString(parameters.layerSizeList.toArray()), 
        parameters.squashingFunctionName, parameters.learningRate, parameters.momemtumWeight, 
        parameters.regularizationWeight);

    return true;
  }

  static Double getDouble(CommandLine commandLine, Option option) {
    Object val = commandLine.getValue(option);
    if (val != null) {
      return Double.parseDouble(val.toString());
    }
    return null;
  }

  static String getString(CommandLine commandLine, Option option) {
    Object val = commandLine.getValue(option);
    if (val != null) {
      return val.toString();
    }
    return null;
  }

  static List<Integer> getIntegerList(CommandLine commandLine, Option option) {
    List<String> list = commandLine.getValues(option);
    List<Integer> valList = Lists.newArrayList();
    for (String str : list) {
      valList.add(Integer.parseInt(str));
    }
    return valList;
  }

  static List<String> getStringList(CommandLine commandLine, Option option) {
    return commandLine.getValues(option);
  }

}