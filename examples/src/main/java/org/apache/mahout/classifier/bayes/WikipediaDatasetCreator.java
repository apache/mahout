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

package org.apache.mahout.classifier.bayes;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;

import java.io.IOException;

public class WikipediaDatasetCreator {
  private WikipediaDatasetCreator() {
  }

  @SuppressWarnings("static-access")
  public static void main(String[] args) throws IOException,
      OptionException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option dirInputPathOpt = obuilder.withLongName("dirInputPath").withRequired(true).withArgument(
            abuilder.withName("dirInputPath").withMinimum(1).withMaximum(1).create()).
            withDescription("The input directory path").withShortName("i").create();

    Option dirOutputPathOpt = obuilder.withLongName("dirOutputPath").withRequired(true).withArgument(
            abuilder.withName("dirOutputPath").withMinimum(1).withMaximum(1).create()).
            withDescription("The output directory Path").withShortName("o").create();

    Option countriesFileOpt = obuilder.withLongName("countriesFile").withRequired(true).withArgument(
            abuilder.withName("countriesFile").withMinimum(1).withMaximum(1).create()).
            withDescription("Location of the countries file").withShortName("c").create();

    Group group = gbuilder.withName("Options").withOption(countriesFileOpt).withOption(dirInputPathOpt).withOption(dirOutputPathOpt).create();

    Parser parser = new Parser();
    parser.setGroup(group);
    CommandLine cmdLine = parser.parse(args);


    String dirInputPath = (String) cmdLine.getValue(dirInputPathOpt);
    String dirOutputPath = (String) cmdLine.getValue(dirOutputPathOpt);
    String countriesFile = (String) cmdLine.getValue(countriesFileOpt);

    WikipediaDatasetCreatorDriver.runJob(dirInputPath, dirOutputPath, countriesFile);

  }
}
