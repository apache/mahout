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

package org.apache.mahout.classifier.sgd;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

import com.google.common.base.Charsets;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.NewsgroupHelper;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.Dictionary;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;

/**
 * Run the 20 news groups test data through SGD, as trained by {@link org.apache.mahout.classifier.sgd.TrainNewsGroups}.
 */
public final class TestNewsGroups {

  private String inputFile;
  private String modelFile;

  private TestNewsGroups() {
  }

  public static void main(String[] args) throws IOException {
    TestNewsGroups runner = new TestNewsGroups();
    if (runner.parseArgs(args)) {
      runner.run(new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
    }
  }

  public void run(PrintWriter output) throws IOException {

    File base = new File(inputFile);
    //contains the best model
    OnlineLogisticRegression classifier =
        ModelSerializer.readBinary(new FileInputStream(modelFile), OnlineLogisticRegression.class);

    Dictionary newsGroups = new Dictionary();
    Multiset<String> overallCounts = HashMultiset.create();

    List<File> files = Lists.newArrayList();
    for (File newsgroup : base.listFiles()) {
      if (newsgroup.isDirectory()) {
        newsGroups.intern(newsgroup.getName());
        files.addAll(Arrays.asList(newsgroup.listFiles()));
      }
    }
    System.out.println(files.size() + " test files");
    ResultAnalyzer ra = new ResultAnalyzer(newsGroups.values(), "DEFAULT");
    for (File file : files) {
      String ng = file.getParentFile().getName();

      int actual = newsGroups.intern(ng);
      NewsgroupHelper helper = new NewsgroupHelper();
      //no leak type ensures this is a normal vector
      Vector input = helper.encodeFeatureVector(file, actual, 0, overallCounts);
      Vector result = classifier.classifyFull(input);
      int cat = result.maxValueIndex();
      double score = result.maxValue();
      double ll = classifier.logLikelihood(actual, input);
      ClassifierResult cr = new ClassifierResult(newsGroups.values().get(cat), score, ll);
      ra.addInstance(newsGroups.values().get(actual), cr);

    }
    output.println(ra);
  }

  boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

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

    inputFile = (String) cmdLine.getValue(inputFileOption);
    modelFile = (String) cmdLine.getValue(modelFileOption);
    return true;
  }

}
