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
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.Parser;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.classifier.BayesFileFormatter;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;


/**
 * Prepare the 20 Newsgroups files for training using the {@link org.apache.mahout.classifier.BayesFileFormatter}.
 *
 * This class takes the directory containing the extracted newsgroups and collapses them into a single file per category, with
 * one document per line (first token on each line is the label) 
 *
 */
public class PrepareTwentyNewsgroups {

  @SuppressWarnings("static-access")
  public static void main(String[] args) throws IOException, ParseException,
      ClassNotFoundException, InstantiationException, IllegalAccessException {
    Options options = new Options();
    Option parentOpt = OptionBuilder.withLongOpt("parent").isRequired().hasArg().withDescription("Parent dir containing the newsgroups").create("p");
    options.addOption(parentOpt);
    Option outputDirOpt = OptionBuilder.withLongOpt("outputDir").isRequired().hasArg().withDescription("The output directory").create("o");
    options.addOption(outputDirOpt);
    Option analyzerNameOpt = OptionBuilder.withLongOpt("analyzerName").isRequired().hasArg().withDescription("The class name of the analyzer").create("a");
    options.addOption(analyzerNameOpt);
    Option charsetOpt = OptionBuilder.withLongOpt("charset").hasArg().isRequired().withDescription("The name of the character encoding of the input files").create("c");
    options.addOption(charsetOpt);

    Parser parser = new PosixParser();
    CommandLine cmdLine = parser.parse(options, args);

    File parentDir = new File(cmdLine.getOptionValue(parentOpt.getOpt()));
    File outputDir = new File(cmdLine.getOptionValue(outputDirOpt.getOpt()));
    String analyzerName = cmdLine.getOptionValue(analyzerNameOpt.getOpt());
    Charset charset = Charset.forName(cmdLine.getOptionValue(charsetOpt.getOpt()));
    Analyzer analyzer = (Analyzer) Class.forName(analyzerName).newInstance();
    //parent dir contains dir by category
    File [] categoryDirs = parentDir.listFiles();
    for (File dir : categoryDirs) {
      if (dir.isDirectory()){
        File outputFile = new File(outputDir, dir.getName() + ".txt");
        BayesFileFormatter.collapse(dir.getName(), analyzer, dir, charset, outputFile);
      }
    }
  }
}