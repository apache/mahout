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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression.Wrapper;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Resources;

public final class TrainAdaptiveLogistic {

  private static String inputFile;
  private static String outputFile;
  private static AdaptiveLogisticModelParameters lmp;
  private static int passes;
  private static boolean showperf;
  private static int skipperfnum = 99;
  private static AdaptiveLogisticRegression model;

  private TrainAdaptiveLogistic() {
  }

  public static void main(String[] args) throws Exception {
    mainToOutput(args, new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
  }

  static void mainToOutput(String[] args, PrintWriter output) throws Exception {
    if (parseArgs(args)) {

      CsvRecordFactory csv = lmp.getCsvRecordFactory();
      model = lmp.createAdaptiveLogisticRegression();
      State<Wrapper, CrossFoldLearner> best;
      CrossFoldLearner learner = null;

      int k = 0;
      for (int pass = 0; pass < passes; pass++) {
        BufferedReader in = open(inputFile);

        // read variable names
        csv.firstLine(in.readLine());

        String line = in.readLine();
        while (line != null) {
          // for each new line, get target and predictors
          Vector input = new RandomAccessSparseVector(lmp.getNumFeatures());
          int targetValue = csv.processLine(line, input);

          // update model
          model.train(targetValue, input);
          k++;

          if (showperf && (k % (skipperfnum + 1) == 0)) {

            best = model.getBest();
            if (best != null) {
              learner = best.getPayload().getLearner();
            }
            if (learner != null) {
              double averageCorrect = learner.percentCorrect();
              double averageLL = learner.logLikelihood();
              output.printf("%d\t%.3f\t%.2f%n",
                            k, averageLL, averageCorrect * 100);
            } else {
              output.printf(Locale.ENGLISH,
                            "%10d %2d %s%n", k, targetValue,
                            "AdaptiveLogisticRegression has not found a good model ......");
            }
          }
          line = in.readLine();
        }
        in.close();
      }

      best = model.getBest();
      if (best != null) {
        learner = best.getPayload().getLearner();
      }
      if (learner == null) {
        output.println("AdaptiveLogisticRegression has failed to train a model.");
        return;
      }


      OutputStream modelOutput = new FileOutputStream(outputFile);
      try {
        lmp.saveTo(modelOutput);
      } finally {
        modelOutput.close();
      }

      OnlineLogisticRegression lr = learner.getModels().get(0);
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

    Option help = builder.withLongName("help")
        .withDescription("print this list").create();

    Option quiet = builder.withLongName("quiet")
        .withDescription("be extra quiet").create();
    
   
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option showperf = builder
      .withLongName("showperf")
      .withDescription("output performance measures during training")
      .create();

    Option inputFile = builder
        .withLongName("input")
        .withRequired(true)
        .withArgument(
            argumentBuilder.withName("input").withMaximum(1)
                .create())
        .withDescription("where to get training data").create();

    Option outputFile = builder
        .withLongName("output")
        .withRequired(true)
        .withArgument(
            argumentBuilder.withName("output").withMaximum(1)
                .create())
        .withDescription("where to write the model content").create();

    Option threads = builder.withLongName("threads")
        .withArgument(
            argumentBuilder.withName("threads").withDefault("4").create())
        .withDescription("the number of threads AdaptiveLogisticRegression uses")
        .create();


    Option predictors = builder.withLongName("predictors")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("predictors").create())
        .withDescription("a list of predictor variables").create();

    Option types = builder
        .withLongName("types")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("types").create())
        .withDescription(
            "a list of predictor variable types (numeric, word, or text)")
        .create();

    Option target = builder
        .withLongName("target")
        .withDescription("the name of the target variable")    
        .withRequired(true)    
        .withArgument(
            argumentBuilder.withName("target").withMaximum(1)
                .create())
         .create();
    
    Option targetCategories = builder
      .withLongName("categories")
      .withDescription("the number of target categories to be considered")
      .withRequired(true)
      .withArgument(argumentBuilder.withName("categories").withMaximum(1).create())
      .create();
    

    Option features = builder
        .withLongName("features")
        .withDescription("the number of internal hashed features to use")
        .withArgument(
            argumentBuilder.withName("numFeatures")
                .withDefault("1000").withMaximum(1).create())        
        .create();

    Option passes = builder
        .withLongName("passes")
        .withDescription("the number of times to pass over the input data")
        .withArgument(
            argumentBuilder.withName("passes").withDefault("2")
                .withMaximum(1).create())        
        .create();

    Option interval = builder.withLongName("interval")
        .withArgument(
            argumentBuilder.withName("interval").withDefault("500").create())
        .withDescription("the interval property of AdaptiveLogisticRegression")
        .create();

    Option window = builder.withLongName("window")
        .withArgument(
            argumentBuilder.withName("window").withDefault("800").create())
        .withDescription("the average propery of AdaptiveLogisticRegression")
        .create();

    Option skipperfnum = builder.withLongName("skipperfnum")
        .withArgument(
            argumentBuilder.withName("skipperfnum").withDefault("99").create())
        .withDescription("show performance measures every (skipperfnum + 1) rows")
        .create();

    Option prior = builder.withLongName("prior")
        .withArgument(
            argumentBuilder.withName("prior").withDefault("L1").create())
        .withDescription("the prior algorithm to use: L1, L2, ebp, tp, up")
        .create();

    Option priorOption = builder.withLongName("prioroption")
        .withArgument(
            argumentBuilder.withName("prioroption").create())
        .withDescription("constructor parameter for ElasticBandPrior and TPrior")
        .create();

    Option auc = builder.withLongName("auc")
        .withArgument(
            argumentBuilder.withName("auc").withDefault("global").create())
        .withDescription("the auc to use: global or grouped")
        .create();

    

    Group normalArgs = new GroupBuilder().withOption(help)
        .withOption(quiet).withOption(inputFile).withOption(outputFile)
        .withOption(target).withOption(targetCategories)
        .withOption(predictors).withOption(types).withOption(passes)
        .withOption(interval).withOption(window).withOption(threads)
        .withOption(prior).withOption(features).withOption(showperf)
        .withOption(skipperfnum).withOption(priorOption).withOption(auc)
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

    TrainAdaptiveLogistic.inputFile = getStringArgument(cmdLine, inputFile);
    TrainAdaptiveLogistic.outputFile = getStringArgument(cmdLine,
                                                         outputFile);

    List<String> typeList = Lists.newArrayList();
    for (Object x : cmdLine.getValues(types)) {
      typeList.add(x.toString());
    }

    List<String> predictorList = Lists.newArrayList();
    for (Object x : cmdLine.getValues(predictors)) {
      predictorList.add(x.toString());
    }

    lmp = new AdaptiveLogisticModelParameters();
    lmp.setTargetVariable(getStringArgument(cmdLine, target));
    lmp.setMaxTargetCategories(getIntegerArgument(cmdLine, targetCategories));
    lmp.setNumFeatures(getIntegerArgument(cmdLine, features));
    lmp.setInterval(getIntegerArgument(cmdLine, interval));
    lmp.setAverageWindow(getIntegerArgument(cmdLine, window));
    lmp.setThreads(getIntegerArgument(cmdLine, threads));
    lmp.setAuc(getStringArgument(cmdLine, auc));
    lmp.setPrior(getStringArgument(cmdLine, prior));
    if (cmdLine.getValue(priorOption) != null) {
      lmp.setPriorOption(getDoubleArgument(cmdLine, priorOption));
    }
    lmp.setTypeMap(predictorList, typeList);
    TrainAdaptiveLogistic.showperf = getBooleanArgument(cmdLine, showperf);
    TrainAdaptiveLogistic.skipperfnum = getIntegerArgument(cmdLine, skipperfnum);
    TrainAdaptiveLogistic.passes = getIntegerArgument(cmdLine, passes);

    lmp.checkParameters();

    return true;
  }

  private static String getStringArgument(CommandLine cmdLine,
                                          Option inputFile) {
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

  public static AdaptiveLogisticRegression getModel() {
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
