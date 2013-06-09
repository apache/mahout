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

package org.apache.mahout.classifier.sgd;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Locale;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import com.google.common.io.Resources;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;


/**
 * Train a logistic regression for the examples from Chapter 13 of Mahout in Action
 */
public final class TrainLogistic {

  private static String inputFile;
  private static String outputFile;
  private static LogisticModelParameters lmp;
  private static int passes;
  private static boolean scores;
  private static OnlineLogisticRegression model;

  private TrainLogistic() {
  }

  public static void main(String[] args) throws Exception {
    mainToOutput(args, new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
  }

  static void mainToOutput(String[] args, PrintWriter output) throws Exception {
    if (parseArgs(args)) {
      double logPEstimate = 0;
      int samples = 0;

      CsvRecordFactory csv = lmp.getCsvRecordFactory();
      OnlineLogisticRegression lr = lmp.createRegression();
      for (int pass = 0; pass < passes; pass++) {
        BufferedReader in = open(inputFile);
        try {
          // read variable names
          csv.firstLine(in.readLine());

          String line = in.readLine();
          while (line != null) {
            // for each new line, get target and predictors
            Vector input = new RandomAccessSparseVector(lmp.getNumFeatures());
            int targetValue = csv.processLine(line, input);

            // check performance while this is still news
            double logP = lr.logLikelihood(targetValue, input);
            if (!Double.isInfinite(logP)) {
              if (samples < 20) {
                logPEstimate = (samples * logPEstimate + logP) / (samples + 1);
              } else {
                logPEstimate = 0.95 * logPEstimate + 0.05 * logP;
              }
              samples++;
            }
            double p = lr.classifyScalar(input);
            if (scores) {
              output.printf(Locale.ENGLISH, "%10d %2d %10.2f %2.4f %10.4f %10.4f%n",
                samples, targetValue, lr.currentLearningRate(), p, logP, logPEstimate);
            }

            // now update model
            lr.train(targetValue, input);

            line = in.readLine();
          }
        } finally {
          Closeables.close(in, true);
        }
      }

      OutputStream modelOutput = new FileOutputStream(outputFile);
      try {
        lmp.saveTo(modelOutput);
      } finally {
        Closeables.close(modelOutput, false);
      }

      output.println(lmp.getNumFeatures());
      output.println(lmp.getTargetVariable() + " ~ ");
      String sep = "";
      for (String v : csv.getTraceDictionary().keySet()) {
        double weight = predictorWeight(lr, 0, csv, v);
        if (weight != 0) {
          output.printf(Locale.ENGLISH, "%s%.3f*%s", sep, weight, v);
          sep = " + ";
        }
      }
      output.printf("%n");
      model = lr;
      for (int row = 0; row < lr.getBeta().numRows(); row++) {
        for (String key : csv.getTraceDictionary().keySet()) {
          double weight = predictorWeight(lr, row, csv, key);
          if (weight != 0) {
            output.printf(Locale.ENGLISH, "%20s %.5f%n", key, weight);
          }
        }
        for (int column = 0; column < lr.getBeta().numCols(); column++) {
          output.printf(Locale.ENGLISH, "%15.9f ", lr.getBeta().get(row, column));
        }
        output.println();
      }
    }
  }

  private static double predictorWeight(OnlineLogisticRegression lr, int row, RecordFactory csv, String predictor) {
    double weight = 0;
    for (Integer column : csv.getTraceDictionary().get(predictor)) {
      weight += lr.getBeta().get(row, column);
    }
    return weight;
  }

  private static boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    Option quiet = builder.withLongName("quiet").withDescription("be extra quiet").create();
    Option scores = builder.withLongName("scores").withDescription("output score diagnostics during training").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFile = builder.withLongName("input")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
            .withDescription("where to get training data")
            .create();

    Option outputFile = builder.withLongName("output")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
            .withDescription("where to get training data")
            .create();

    Option predictors = builder.withLongName("predictors")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("p").create())
            .withDescription("a list of predictor variables")
            .create();

    Option types = builder.withLongName("types")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("t").create())
            .withDescription("a list of predictor variable types (numeric, word, or text)")
            .create();

    Option target = builder.withLongName("target")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("target").withMaximum(1).create())
            .withDescription("the name of the target variable")
            .create();

    Option features = builder.withLongName("features")
            .withArgument(
                    argumentBuilder.withName("numFeatures")
                            .withDefault("1000")
                            .withMaximum(1).create())
            .withDescription("the number of internal hashed features to use")
            .create();

    Option passes = builder.withLongName("passes")
            .withArgument(
                    argumentBuilder.withName("passes")
                            .withDefault("2")
                            .withMaximum(1).create())
            .withDescription("the number of times to pass over the input data")
            .create();

    Option lambda = builder.withLongName("lambda")
            .withArgument(argumentBuilder.withName("lambda").withDefault("1e-4").withMaximum(1).create())
            .withDescription("the amount of coefficient decay to use")
            .create();

    Option rate = builder.withLongName("rate")
            .withArgument(argumentBuilder.withName("learningRate").withDefault("1e-3").withMaximum(1).create())
            .withDescription("the learning rate")
            .create();

    Option noBias = builder.withLongName("noBias")
            .withDescription("don't include a bias term")
            .create();

    Option targetCategories = builder.withLongName("categories")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("number").withMaximum(1).create())
            .withDescription("the number of target categories to be considered")
            .create();

    Group normalArgs = new GroupBuilder()
            .withOption(help)
            .withOption(quiet)
            .withOption(inputFile)
            .withOption(outputFile)
            .withOption(target)
            .withOption(targetCategories)
            .withOption(predictors)
            .withOption(types)
            .withOption(passes)
            .withOption(lambda)
            .withOption(rate)
            .withOption(noBias)
            .withOption(features)
            .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
    CommandLine cmdLine = parser.parseAndHelp(args);

    if (cmdLine == null) {
      return false;
    }

    TrainLogistic.inputFile = getStringArgument(cmdLine, inputFile);
    TrainLogistic.outputFile = getStringArgument(cmdLine, outputFile);

    List<String> typeList = Lists.newArrayList();
    for (Object x : cmdLine.getValues(types)) {
      typeList.add(x.toString());
    }

    List<String> predictorList = Lists.newArrayList();
    for (Object x : cmdLine.getValues(predictors)) {
      predictorList.add(x.toString());
    }

    lmp = new LogisticModelParameters();
    lmp.setTargetVariable(getStringArgument(cmdLine, target));
    lmp.setMaxTargetCategories(getIntegerArgument(cmdLine, targetCategories));
    lmp.setNumFeatures(getIntegerArgument(cmdLine, features));
    lmp.setUseBias(!getBooleanArgument(cmdLine, noBias));
    lmp.setTypeMap(predictorList, typeList);

    lmp.setLambda(getDoubleArgument(cmdLine, lambda));
    lmp.setLearningRate(getDoubleArgument(cmdLine, rate));

    TrainLogistic.scores = getBooleanArgument(cmdLine, scores);
    TrainLogistic.passes = getIntegerArgument(cmdLine, passes);

    return true;
  }

  private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
    return (String) cmdLine.getValue(inputFile);
  }

  private static boolean getBooleanArgument(CommandLine cmdLine, Option option) {
    return cmdLine.hasOption(option);
  }

  private static int getIntegerArgument(CommandLine cmdLine, Option features) {
    return Integer.parseInt((String) cmdLine.getValue(features));
  }

  private static double getDoubleArgument(CommandLine cmdLine, Option op) {
    return Double.parseDouble((String) cmdLine.getValue(op));
  }

  public static OnlineLogisticRegression getModel() {
    return model;
  }

  public static LogisticModelParameters getParameters() {
    return lmp;
  }

  static BufferedReader open(String inputFile) throws IOException {
    InputStream in;
    try {
      in = Resources.getResource(inputFile).openStream();
    } catch (IllegalArgumentException e) {
      in = new FileInputStream(new File(inputFile));
    }
    return new BufferedReader(new InputStreamReader(in, Charsets.UTF_8));
  }
}
