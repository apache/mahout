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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.io.Charsets;
import org.apache.mahout.common.CommandLineUtil;

/**
 * Command-line tool for generating random sequences by given HMM
 */
public final class RandomSequenceGenerator {

  private RandomSequenceGenerator() {
  }

  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    Option outputOption = optionBuilder.withLongName("output").
      withDescription("Output file with sequence of observed states").
      withShortName("o").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(false).create();

    Option modelOption = optionBuilder.withLongName("model").
      withDescription("Path to serialized HMM model").
      withShortName("m").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(true).create();

    Option lengthOption = optionBuilder.withLongName("length").
      withDescription("Length of generated sequence").
      withShortName("l").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    Group optionGroup = new GroupBuilder().
      withOption(outputOption).withOption(modelOption).withOption(lengthOption).
      withName("Options").create();

    try {
      Parser parser = new Parser();
      parser.setGroup(optionGroup);
      CommandLine commandLine = parser.parse(args);

      String output = (String) commandLine.getValue(outputOption);

      String modelPath = (String) commandLine.getValue(modelOption);

      int length = Integer.parseInt((String) commandLine.getValue(lengthOption));

      //reading serialized HMM
      HmmModel model;
      try (DataInputStream modelStream = new DataInputStream(new FileInputStream(modelPath))){
        model = LossyHmmSerializer.deserialize(modelStream);
      }

      //generating observations
      int[] observations = HmmEvaluator.predict(model, length, System.currentTimeMillis());

      //writing output
      try (PrintWriter writer =
               new PrintWriter(new OutputStreamWriter(new FileOutputStream(output), Charsets.UTF_8), true)){
        for (int observation : observations) {
          writer.print(observation);
          writer.print(' ');
        }
      }
    } catch (OptionException e) {
      CommandLineUtil.printHelp(optionGroup);
    }
  }
}
