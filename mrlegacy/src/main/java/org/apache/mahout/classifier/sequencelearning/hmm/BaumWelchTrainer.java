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

import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Date;
import java.util.List;
import java.util.Scanner;

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
 * A class for EM training of HMM from console
 */
public final class BaumWelchTrainer {

  private BaumWelchTrainer() {
  }

  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    Option inputOption = DefaultOptionCreator.inputOption().create();

    Option outputOption = DefaultOptionCreator.outputOption().create();

    Option stateNumberOption = optionBuilder.withLongName("nrOfHiddenStates").
      withDescription("Number of hidden states").
      withShortName("nh").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    Option observedStateNumberOption = optionBuilder.withLongName("nrOfObservedStates").
      withDescription("Number of observed states").
      withShortName("no").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    Option epsilonOption = optionBuilder.withLongName("epsilon").
      withDescription("Convergence threshold").
      withShortName("e").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    Option iterationsOption = optionBuilder.withLongName("max-iterations").
      withDescription("Maximum iterations number").
      withShortName("m").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    Group optionGroup = new GroupBuilder().withOption(inputOption).
      withOption(outputOption).withOption(stateNumberOption).withOption(observedStateNumberOption).
      withOption(epsilonOption).withOption(iterationsOption).
      withName("Options").create();

    try {
      Parser parser = new Parser();
      parser.setGroup(optionGroup);
      CommandLine commandLine = parser.parse(args);

      String input = (String) commandLine.getValue(inputOption);
      String output = (String) commandLine.getValue(outputOption);

      int nrOfHiddenStates = Integer.parseInt((String) commandLine.getValue(stateNumberOption));
      int nrOfObservedStates = Integer.parseInt((String) commandLine.getValue(observedStateNumberOption));

      double epsilon = Double.parseDouble((String) commandLine.getValue(epsilonOption));
      int maxIterations = Integer.parseInt((String) commandLine.getValue(iterationsOption));

      //constructing random-generated HMM
      HmmModel model = new HmmModel(nrOfHiddenStates, nrOfObservedStates, new Date().getTime());
      List<Integer> observations = Lists.newArrayList();

      //reading observations
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

      //training
      HmmModel trainedModel = HmmTrainer.trainBaumWelch(model,
        observationsArray, epsilon, maxIterations, true);

      //serializing trained model
      DataOutputStream stream  = new DataOutputStream(new FileOutputStream(output));
      try {
        LossyHmmSerializer.serialize(trainedModel, stream);
      } finally {
        Closeables.close(stream, false);
      }

      //printing tranied model
      System.out.println("Initial probabilities: ");
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i) {
        System.out.print(i + " ");
      }
      System.out.println();
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i) {
        System.out.print(trainedModel.getInitialProbabilities().get(i) + " ");
      }
      System.out.println();

      System.out.println("Transition matrix:");
      System.out.print("  ");
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i) {
        System.out.print(i + " ");
      }
      System.out.println();
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i) {
        System.out.print(i + " ");
        for (int j = 0; j < trainedModel.getNrOfHiddenStates(); ++j) {
          System.out.print(trainedModel.getTransitionMatrix().get(i, j) + " ");
        }
        System.out.println();
      }
      System.out.println("Emission matrix: ");
      System.out.print("  ");
      for (int i = 0; i < trainedModel.getNrOfOutputStates(); ++i) {
        System.out.print(i + " ");
      }
      System.out.println();
      for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i) {
        System.out.print(i + " ");
        for (int j = 0; j < trainedModel.getNrOfOutputStates(); ++j) {
          System.out.print(trainedModel.getEmissionMatrix().get(i, j) + " ");
        }
        System.out.println();
      }
    } catch (OptionException e) {
      CommandLineUtil.printHelp(optionGroup);
    }
  }
}
