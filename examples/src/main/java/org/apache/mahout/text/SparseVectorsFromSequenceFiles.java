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

package org.apache.mahout.text;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.utils.vectors.text.DictionaryVectorizer;

/**
 * Converts a given set of sequence files into SparseVectors
 * 
 */
public final class SparseVectorsFromSequenceFiles {
  
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputDirOpt =
        obuilder.withLongName("input").withRequired(true).withArgument(
            abuilder.withName("input").withMinimum(1).withMaximum(1).create())
            .withDescription(
                "input dir containing the documents in sequence file format")
            .withShortName("i").create();
    
    Option outputDirOpt =
        obuilder.withLongName("outputDir").withRequired(true).withArgument(
            abuilder.withName("outputDir").withMinimum(1).withMaximum(1)
                .create()).withDescription("The output directory")
            .withShortName("o").create();
    Option minSupportOpt =
        obuilder.withLongName("minSupport").withArgument(
            abuilder.withName("minSupport").withMinimum(1).withMaximum(1)
                .create()).withDescription(
            "(Optional) Minimum Support. Default Value: 2").withShortName("s")
            .create();
    
    Option analyzerNameOpt =
        obuilder.withLongName("analyzerName").withArgument(
            abuilder.withName("analyzerName").withMinimum(1).withMaximum(1)
                .create()).withDescription("The class name of the analyzer")
            .withShortName("a").create();
    
    Option chunkSizeOpt =
        obuilder.withLongName("chunkSize").withArgument(
            abuilder.withName("chunkSize").withMinimum(1).withMaximum(1)
                .create()).withDescription(
            "The chunkSize in MegaBytes. 100-10000 MB").withShortName("chunk")
            .create();
    
    Group group =
        gbuilder.withName("Options").withOption(minSupportOpt).withOption(
            analyzerNameOpt).withOption(chunkSizeOpt).withOption(outputDirOpt)
            .withOption(inputDirOpt).create();
    
    Parser parser = new Parser();
    parser.setGroup(group);
    CommandLine cmdLine = parser.parse(args);
    
    String inputDir = (String) cmdLine.getValue(inputDirOpt);
    String outputDir = (String) cmdLine.getValue(outputDirOpt);
    
    int chunkSize = 100;
    if (cmdLine.hasOption(chunkSizeOpt)) {
      chunkSize =
          Integer.valueOf((String) cmdLine.getValue(chunkSizeOpt)).intValue();
    }
    int minSupport = 2;
    if (cmdLine.hasOption(minSupportOpt)) {
      String minSupportString = (String) cmdLine.getValue(minSupportOpt);
      minSupport = Integer.parseInt(minSupportString);
    }
    String analyzerName = (String) cmdLine.getValue(analyzerNameOpt);
    Class<? extends Analyzer> analyzerClass = StandardAnalyzer.class;
    if (cmdLine.hasOption(analyzerNameOpt)) {
      String className = cmdLine.getValue(analyzerNameOpt).toString();
      analyzerClass = (Class<? extends Analyzer>) Class.forName(className);
      // try instantiating it, b/c there isn't any point in setting it if
      // you can't instantiate it
      analyzerClass.newInstance();
    }
    DictionaryVectorizer.createTermFrequencyVectors(inputDir, outputDir,
        analyzerClass, minSupport, chunkSize);
  }
}
