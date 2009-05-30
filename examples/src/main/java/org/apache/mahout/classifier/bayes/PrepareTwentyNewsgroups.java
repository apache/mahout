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

import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.classifier.BayesFileFormatter;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;

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

  private PrepareTwentyNewsgroups() {
  }

  public static void main(String[] args) throws IOException,
          ClassNotFoundException, InstantiationException, IllegalAccessException, OptionException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option parentOpt = obuilder.withLongName("parent").withRequired(true).withArgument(
            abuilder.withName("parent").withMinimum(1).withMaximum(1).create()).
            withDescription("Parent dir containing the newsgroups").withShortName("p").create();

    Option outputDirOpt = obuilder.withLongName("outputDir").withRequired(true).withArgument(
            abuilder.withName("outputDir").withMinimum(1).withMaximum(1).create()).
            withDescription("The output directory").withShortName("o").create();

    Option analyzerNameOpt = obuilder.withLongName("analyzerName").withRequired(true).withArgument(
            abuilder.withName("analyzerName").withMinimum(1).withMaximum(1).create()).
            withDescription("The class name of the analyzer").withShortName("a").create();

    Option charsetOpt = obuilder.withLongName("charset").withRequired(true).withArgument(
            abuilder.withName("charset").withMinimum(1).withMaximum(1).create()).
            withDescription("The name of the character encoding of the input files").withShortName("c").create();

    Group group = gbuilder.withName("Options").withOption(analyzerNameOpt).withOption(charsetOpt).withOption(outputDirOpt).withOption(parentOpt).create();

    Parser parser = new Parser();
    parser.setGroup(group);
    CommandLine cmdLine = parser.parse(args);


    File parentDir = new File((String) cmdLine.getValue(parentOpt));
    File outputDir = new File((String) cmdLine.getValue(outputDirOpt));
    String analyzerName = (String) cmdLine.getValue(analyzerNameOpt);
    Charset charset = Charset.forName((String) cmdLine.getValue(charsetOpt));
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