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
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Locale;

import com.google.common.base.Charsets;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.mahout.classifier.ConfusionMatrix;
import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression.Wrapper;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.stats.OnlineSummarizer;

/*
 * Auc and averageLikelihood are always shown if possible, if the number of target value is more than 2, 
 * then Auc and entropy matirx are not shown regardless the value of showAuc and showEntropy
 * the user passes, because the current implementation does not support them on two value targets.
 * */
public final class ValidateAdaptiveLogistic {

  private static String inputFile;
  private static String modelFile;
  private static String defaultCategory;
  private static boolean showAuc;
  private static boolean showScores;
  private static boolean showConfusion;

  private ValidateAdaptiveLogistic() {
  }

  public static void main(String[] args) throws IOException {
    mainToOutput(args, new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
  }

  static void mainToOutput(String[] args, PrintWriter output) throws IOException {
    if (parseArgs(args)) {
      if (!showAuc && !showConfusion && !showScores) {
        showAuc = true;
        showConfusion = true;
      }

      Auc collector = null;
      AdaptiveLogisticModelParameters lmp = AdaptiveLogisticModelParameters
          .loadFromFile(new File(modelFile));
      CsvRecordFactory csv = lmp.getCsvRecordFactory();
      AdaptiveLogisticRegression lr = lmp.createAdaptiveLogisticRegression();      

      if (lmp.getTargetCategories().size() <= 2) {
        collector = new Auc();
      }

      OnlineSummarizer slh = new OnlineSummarizer();
      ConfusionMatrix cm = new ConfusionMatrix(lmp.getTargetCategories(), defaultCategory);

      State<Wrapper, CrossFoldLearner> best = lr.getBest();
      if (best == null) {
        output.println("AdaptiveLogisticRegression has not be trained probably.");
        return;
      }
      CrossFoldLearner learner = best.getPayload().getLearner();

      BufferedReader in = TrainLogistic.open(inputFile);
      String line = in.readLine();
      csv.firstLine(line);
      line = in.readLine();
      if (showScores) {
        output.println("\"target\", \"model-output\", \"log-likelihood\", \"average-likelihood\"");
      }
      while (line != null) {
        Vector v = new SequentialAccessSparseVector(lmp.getNumFeatures());
        //TODO: How to avoid extra target values not shown in the training process.
        int target = csv.processLine(line, v);
        double likelihood = learner.logLikelihood(target, v);
        double score = learner.classifyFull(v).maxValue();

        slh.add(likelihood);
        cm.addInstance(csv.getTargetString(line), csv.getTargetLabel(target));        

        if (showScores) {
          output.printf(Locale.ENGLISH, "%8d, %.12f, %.13f, %.13f%n", target,
              score, learner.logLikelihood(target, v), slh.getMean());
        }
        if (collector != null) {
          collector.add(target, score);
        }
        line = in.readLine();
      }

      output.printf(Locale.ENGLISH,"\nLog-likelihood:");
      output.printf(Locale.ENGLISH, "Min=%.2f, Max=%.2f, Mean=%.2f, Median=%.2f%n",
          slh.getMin(), slh.getMax(), slh.getMean(), slh.getMedian());

      if (collector != null) {        
        output.printf(Locale.ENGLISH, "%nAUC = %.2f%n", collector.auc());
      }

      if (showConfusion) {
        output.printf(Locale.ENGLISH, "%n%s%n%n", cm.toString());

        if (collector != null) {
          Matrix m = collector.entropy();
          output.printf(Locale.ENGLISH,
              "Entropy Matrix: [[%.1f, %.1f], [%.1f, %.1f]]%n", m.get(0, 0),
              m.get(1, 0), m.get(0, 1), m.get(1, 1));
        }        
      }

    }
  }

  private static boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help")
        .withDescription("print this list").create();

    Option quiet = builder.withLongName("quiet")
        .withDescription("be extra quiet").create();

    Option auc = builder.withLongName("auc").withDescription("print AUC")
        .create();
    Option confusion = builder.withLongName("confusion")
        .withDescription("print confusion matrix").create();

    Option scores = builder.withLongName("scores")
        .withDescription("print scores").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFileOption = builder
        .withLongName("input")
        .withRequired(true)
        .withArgument(
            argumentBuilder.withName("input").withMaximum(1)
                .create())
        .withDescription("where to get validate data").create();

    Option modelFileOption = builder
        .withLongName("model")
        .withRequired(true)
        .withArgument(
            argumentBuilder.withName("model").withMaximum(1)
                .create())
        .withDescription("where to get the trained model").create();

    Option defaultCagetoryOption = builder
      .withLongName("defaultCategory")
      .withRequired(false)
      .withArgument(
          argumentBuilder.withName("defaultCategory").withMaximum(1).withDefault("unknown")
          .create())
      .withDescription("the default category value to use").create();

    Group normalArgs = new GroupBuilder().withOption(help)
        .withOption(quiet).withOption(auc).withOption(scores)
        .withOption(confusion).withOption(inputFileOption)
        .withOption(modelFileOption).withOption(defaultCagetoryOption).create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
    CommandLine cmdLine = parser.parseAndHelp(args);

    if (cmdLine == null) {
      return false;
    }

    inputFile = getStringArgument(cmdLine, inputFileOption);
    modelFile = getStringArgument(cmdLine, modelFileOption);
    defaultCategory = getStringArgument(cmdLine, defaultCagetoryOption);
    showAuc = getBooleanArgument(cmdLine, auc);
    showScores = getBooleanArgument(cmdLine, scores);
    showConfusion = getBooleanArgument(cmdLine, confusion);

    return true;
  }

  private static boolean getBooleanArgument(CommandLine cmdLine, Option option) {
    return cmdLine.hasOption(option);
  }

  private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
    return (String) cmdLine.getValue(inputFile);
  }

}
