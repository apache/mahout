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

package org.apache.mahout.classifier.sequencelearning.hmm;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Scanner;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

/**
 * Command-line tool for Viterbi evaluating
 */
public final class ViterbiEvaluator {

  private ViterbiEvaluator() {
  }

  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    Option inputOption = DefaultOptionCreator.inputOption().create();

    Option outputOption = DefaultOptionCreator.outputOption().create();

    Option modelOption = optionBuilder.withLongName("model").
      withDescription("Path to serialized HMM model").
      withShortName("m").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(true).create();

    Option likelihoodOption = optionBuilder.withLongName("likelihood").
      withDescription("Compute likelihood of observed sequence").
      withShortName("l").withRequired(false).create();

    Group optionGroup = new GroupBuilder().withOption(inputOption).
      withOption(outputOption).withOption(modelOption).withOption(likelihoodOption).
      withName("Options").create();

    try {
      Parser parser = new Parser();
      parser.setGroup(optionGroup);
      CommandLine commandLine = parser.parse(args);

      String input = (String) commandLine.getValue(inputOption);
      String output = (String) commandLine.getValue(outputOption);

      String modelPath = (String) commandLine.getValue(modelOption);

      boolean computeLikelihood = commandLine.hasOption(likelihoodOption);

      //reading serialized HMM
      DataInputStream modelStream = new DataInputStream(new FileInputStream(modelPath));
      HmmModel model;
      try {
        model = LossyHmmSerializer.deserialize(modelStream);
      } finally {
        Closeables.close(modelStream, true);
      }

      //reading observations
      List<Integer> observations = Lists.newArrayList();
      Scanner scanner = new Scanner(new FileInputStream(input), "UTF-8");
      try {
        while (scanner.hasNextInt()) {
          observations.add(scanner.nextInt());
        }
      } finally {
        scanner.close();
      }

      int[] observationsArray = new int[observations.size()];
      for (int i = 0; i < observations.size(); ++i) {
        observationsArray[i] = observations.get(i);
      }

      //decoding
      int[] hiddenStates = HmmEvaluator.decode(model, observationsArray, true);

      //writing output
      PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(output), Charsets.UTF_8), true);
      try {
        for (int hiddenState : hiddenStates) {
          writer.print(hiddenState);
          writer.print(' ');
        }
      } finally {
        Closeables.close(writer, false);
      }

      if (computeLikelihood) {
        System.out.println("Likelihood: " + HmmEvaluator.modelLikelihood(model, observationsArray, true));
      }
    } catch (OptionException e) {
      CommandLineUtil.printHelp(optionGroup);
    }
  }
}
