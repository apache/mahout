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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;

/**
 *
 */
public class RunLogistic {
  private static final Logger log = LoggerFactory.getLogger(RunLogistic.class);
  private static String inputFile;
  private static String modelFile;
  private static boolean showAuc = false;
  private static boolean showScores = false;
  private static boolean showConfusion = false;

  public static void main(String[] args) throws IOException {
    if (parseArgs(args)) {
      if (!showAuc && !showConfusion && !showScores) {
        showAuc = true;
        showConfusion = true;
      }

      Auc collector = new Auc();
      LogisticModelParameters lmp = LogisticModelParameters.loadFrom(new File(modelFile));

      CsvRecordFactory csv = lmp.getCsvRecordFactory();
      OnlineLogisticRegression lr = lmp.createRegression();
      BufferedReader in = TrainLogistic.InputOpener.open(inputFile);
      String line = in.readLine();
      csv.firstLine(line);
      line = in.readLine();
      if (showScores) {
        System.out.printf("\"%s\",\"%s\",\"%s\"\n", "target", "model-output", "log-likelihood");
      }
      while (line != null) {
        Vector v = new SequentialAccessSparseVector(lmp.getNumFeatures());
        int target = csv.processLine(line, v);
        double score = lr.classifyScalar(v);
        if (showScores) {
          System.out.printf("%d,%.3f,%.6f\n", target, score, lr.logLikelihood(target, v));
        }
        collector.add(target, score);
        line = in.readLine();
      }

      if (showAuc) {
        System.out.printf("AUC = %.2f\n", collector.auc());
      }
      if (showConfusion) {
        Matrix m = collector.confusion();
        System.out.printf("confusion: [[%.1f, %.1f], [%.1f, %.1f]]\n", m.get(0, 0), m.get(1, 0), m.get(0, 1), m.get(1, 1));
        m = collector.entropy();
        System.out.printf("entropy: [[%.1f, %.1f], [%.1f, %.1f]]\n", m.get(0, 0), m.get(1, 0), m.get(0, 1), m.get(1, 1));
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
    Option inputFile = builder.withLongName("input")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
            .withDescription("where to get training data")
            .create();

    Option modelFile = builder.withLongName("model")
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
            .withOption(inputFile)
            .withOption(modelFile)
            .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
    CommandLine cmdLine;
    cmdLine = parser.parseAndHelp(args);

    if (cmdLine == null) {
      return false;
    }

    RunLogistic.inputFile = getStringArgument(cmdLine, inputFile);
    RunLogistic.modelFile = getStringArgument(cmdLine, modelFile);
    RunLogistic.showAuc = getBooleanArgument(cmdLine, auc);
    RunLogistic.showScores = getBooleanArgument(cmdLine, scores);
    RunLogistic.showConfusion = getBooleanArgument(cmdLine, confusion);

    return true;
  }

  private static boolean getBooleanArgument(CommandLine cmdLine, Option option) {
    return cmdLine.hasOption(option);
  }

  private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
    return (String) cmdLine.getValue(inputFile);
  }

}
