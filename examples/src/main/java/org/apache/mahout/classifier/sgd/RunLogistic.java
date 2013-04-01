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

import com.google.common.base.Charsets;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.classifier.evaluation.Auc;

import java.io.BufferedReader;
import java.io.File;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Locale;

public final class RunLogistic {

  private static String inputFile;
  private static String modelFile;
  private static boolean showAuc;
  private static boolean showScores;
  private static boolean showConfusion;

  private RunLogistic() {
  }

  public static void main(String[] args) throws Exception {
    mainToOutput(args, new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
  }

  static void mainToOutput(String[] args, PrintWriter output) throws Exception {
    if (parseArgs(args)) {
      if (!showAuc && !showConfusion && !showScores) {
        showAuc = true;
        showConfusion = true;
      }

      Auc collector = new Auc();
      LogisticModelParameters lmp = LogisticModelParameters.loadFrom(new File(modelFile));

      CsvRecordFactory csv = lmp.getCsvRecordFactory();
      OnlineLogisticRegression lr = lmp.createRegression();
      BufferedReader in = TrainLogistic.open(inputFile);
      String line = in.readLine();
      csv.firstLine(line);
      line = in.readLine();
      if (showScores) {
        output.println("\"target\",\"model-output\",\"log-likelihood\"");
      }
      while (line != null) {
        Vector v = new SequentialAccessSparseVector(lmp.getNumFeatures());
        int target = csv.processLine(line, v);

        double score = lr.classifyScalar(v);
        if (showScores) {
          output.printf(Locale.ENGLISH, "%d,%.3f,%.6f%n", target, score, lr.logLikelihood(target, v));
        }
        collector.add(target, score);
        line = in.readLine();
      }

      if (showAuc) {
        output.printf(Locale.ENGLISH, "AUC = %.2f%n", collector.auc());
      }
      if (showConfusion) {
        Matrix m = collector.confusion();
        output.printf(Locale.ENGLISH, "confusion: [[%.1f, %.1f], [%.1f, %.1f]]%n",
          m.get(0, 0), m.get(1, 0), m.get(0, 1), m.get(1, 1));
        m = collector.entropy();
        output.printf(Locale.ENGLISH, "entropy: [[%.1f, %.1f], [%.1f, %.1f]]%n",
          m.get(0, 0), m.get(1, 0), m.get(0, 1), m.get(1, 1));
      }
    }
  }

  private static boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    Option quiet = builder.withLongName("quiet").withDescription("be extra quiet").create();

    Option auc = builder.withLongName("auc").withDescription("print AUC").create();
    Option confusion = builder.withLongName("confusion").withDescription("print confusion matrix").create();

    Option scores = builder.withLongName("scores").withDescription("print scores").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFileOption = builder.withLongName("input")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
            .withDescription("where to get training data")
            .create();

    Option modelFileOption = builder.withLongName("model")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("model").withMaximum(1).create())
            .withDescription("where to get a model")
            .create();

    Group normalArgs = new GroupBuilder()
            .withOption(help)
            .withOption(quiet)
            .withOption(auc)
            .withOption(scores)
            .withOption(confusion)
            .withOption(inputFileOption)
            .withOption(modelFileOption)
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

    inputFile = getStringArgument(cmdLine, inputFileOption);
    modelFile = getStringArgument(cmdLine, modelFileOption);
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
