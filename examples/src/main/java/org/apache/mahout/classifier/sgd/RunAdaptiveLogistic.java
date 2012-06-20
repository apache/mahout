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
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression.Wrapper;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

public final class RunAdaptiveLogistic {

  private static String inputFile;
  private static String modelFile;
  private static String outputFile;
  private static String idColumn;
  private static boolean maxScoreOnly;

  private RunAdaptiveLogistic() {
  }

  public static void main(String[] args) throws Exception {
    mainToOutput(args, new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
  }

  static void mainToOutput(String[] args, PrintWriter output) throws Exception {
    if (!parseArgs(args)) {
      return;
    }
    AdaptiveLogisticModelParameters lmp = AdaptiveLogisticModelParameters
        .loadFromFile(new File(modelFile));

    CsvRecordFactory csv = lmp.getCsvRecordFactory();
    csv.setIdName(idColumn);

    AdaptiveLogisticRegression lr = lmp.createAdaptiveLogisticRegression();

    State<Wrapper, CrossFoldLearner> best = lr.getBest();
    if (best == null) {
      output.println("AdaptiveLogisticRegression has not be trained probably.");
      return;
    }
    CrossFoldLearner learner = best.getPayload().getLearner();

    BufferedReader in = TrainAdaptiveLogistic.open(inputFile);
    BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), Charsets.UTF_8));

    out.write(idColumn + ",target,score");
    out.newLine();

    String line = in.readLine();
    csv.firstLine(line);
    line = in.readLine();
    Map<String, Double> results = new HashMap<String, Double>();
    int k = 0;
    while (line != null) {
      Vector v = new SequentialAccessSparseVector(lmp.getNumFeatures());
      csv.processLine(line, v, false);
      Vector scores = learner.classifyFull(v);
      results.clear();
      if (maxScoreOnly) {
        results.put(csv.getTargetLabel(scores.maxValueIndex()),
            scores.maxValue());
      } else {
        for (int i = 0; i < scores.size(); i++) {
          results.put(csv.getTargetLabel(i), scores.get(i));
        }
      }

      for (Map.Entry<String,Double> entry : results.entrySet()) {
        out.write(csv.getIdString(line) + ',' + entry.getKey() + ',' + entry.getValue());
        out.newLine();
      }
      k++;
      if (k % 100 == 0) {
        output.println(k + " records processed");
      }
      line = in.readLine();
    }
    out.flush();
    out.close();
    output.println(k + " records processed totally.");
  }

  private static boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help")
      .withDescription("print this list").create();

    Option quiet = builder.withLongName("quiet")
      .withDescription("be extra quiet").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFileOption = builder
      .withLongName("input")
      .withRequired(true)
      .withArgument(
          argumentBuilder.withName("input").withMaximum(1)
            .create())
      .withDescription("where to get training data").create();

    Option modelFileOption = builder
      .withLongName("model")
      .withRequired(true)
      .withArgument(
          argumentBuilder.withName("model").withMaximum(1)
            .create())
      .withDescription("where to get the trained model").create();
    
    Option outputFileOption = builder
      .withLongName("output")
      .withRequired(true)
      .withDescription("the file path to output scores")
      .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
      .create();
    
    Option idColumnOption = builder
      .withLongName("idcolumn")
      .withRequired(true)
      .withDescription("the name of the id column for each record")
      .withArgument(argumentBuilder.withName("idcolumn").withMaximum(1).create())
      .create();
    
    Option maxScoreOnlyOption = builder
      .withLongName("maxscoreonly")
      .withDescription("only output the target label with max scores")
      .create();

    Group normalArgs = new GroupBuilder()
      .withOption(help).withOption(quiet)
      .withOption(inputFileOption).withOption(modelFileOption)
      .withOption(outputFileOption).withOption(idColumnOption)
      .withOption(maxScoreOnlyOption)
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
    outputFile = getStringArgument(cmdLine, outputFileOption);
    idColumn = getStringArgument(cmdLine, idColumnOption);
    maxScoreOnly = getBooleanArgument(cmdLine, maxScoreOnlyOption);    
    return true;
  }

  private static boolean getBooleanArgument(CommandLine cmdLine, Option option) {
    return cmdLine.hasOption(option);
  }

  private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
    return (String) cmdLine.getValue(inputFile);
  }

}
