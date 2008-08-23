package org.apache.mahout.classifier.bayes;

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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;

import java.io.IOException;

/**
 * 
 * 
 */
public class WikipediaDatasetCreator {

  @SuppressWarnings("static-access")
  public static void main(String[] args) throws IOException,
      ClassNotFoundException, IllegalAccessException, InstantiationException {
    Options options = new Options();
    Option dirInputPathOpt = OptionBuilder.withLongOpt("dirInputPath").isRequired().hasArg()
        .withDescription("The input Directory Path").create("i");
    options.addOption(dirInputPathOpt);
    Option dirOutputPathOpt = OptionBuilder.withLongOpt("dirOuputPath").isRequired().hasArg()
        .withDescription("The output Directory Path").create("o");
    options.addOption(dirOutputPathOpt);
    Option countriesFileOpt = OptionBuilder.withLongOpt("countriesFile").isRequired().hasArg()
        .withDescription("Location of the Countries File").create("c");
    options.addOption(countriesFileOpt);
    
    CommandLine cmdLine;
    try {
      PosixParser parser = new PosixParser();
      cmdLine = parser.parse(options, args);

      String dirInputPath = cmdLine.getOptionValue(dirInputPathOpt.getOpt());
      String dirOutputPath = cmdLine.getOptionValue(dirOutputPathOpt.getOpt());
      String countriesFile = cmdLine.getOptionValue(countriesFileOpt.getOpt());

      WikipediaDatasetCreatorDriver.runJob(dirInputPath, dirOutputPath, countriesFile);
    } catch (Exception exp) {
      exp.printStackTrace(System.err);
    }
  }
}
