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

package org.apache.mahout.vectorizer;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.vectorizer.collocations.llr.LLRReducer;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Converts a given set of sequence files into SparseVectors
 */
public final class SparseVectorsFromSequenceFiles {
  
  private static final Logger log = LoggerFactory.getLogger(SparseVectorsFromSequenceFiles.class);
  
  private SparseVectorsFromSequenceFiles() {
  }
  
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputDirOpt = obuilder.withLongName("input").withRequired(true).withArgument(
      abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
      "input dir containing the documents in sequence file format").withShortName("i").create();
    
    Option outputDirOpt = obuilder.withLongName("output").withRequired(true).withArgument(
      abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
      "The output directory").withShortName("o").create();
    
    Option minSupportOpt = obuilder.withLongName("minSupport").withArgument(
      abuilder.withName("minSupport").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) Minimum Support. Default Value: 2").withShortName("s").create();
    
    Option analyzerNameOpt = obuilder.withLongName("analyzerName").withArgument(
      abuilder.withName("analyzerName").withMinimum(1).withMaximum(1).create()).withDescription(
      "The class name of the analyzer").withShortName("a").create();
    
    Option chunkSizeOpt = obuilder.withLongName("chunkSize").withArgument(
      abuilder.withName("chunkSize").withMinimum(1).withMaximum(1).create()).withDescription(
      "The chunkSize in MegaBytes. 100-10000 MB").withShortName("chunk").create();
    
    Option weightOpt = obuilder.withLongName("weight").withRequired(false).withArgument(
      abuilder.withName("weight").withMinimum(1).withMaximum(1).create()).withDescription(
      "The kind of weight to use. Currently TF or TFIDF").withShortName("wt").create();
    
    Option minDFOpt = obuilder.withLongName("minDF").withRequired(false).withArgument(
      abuilder.withName("minDF").withMinimum(1).withMaximum(1).create()).withDescription(
      "The minimum document frequency.  Default is 1").withShortName("md").create();
    
    Option maxDFPercentOpt = obuilder.withLongName("maxDFPercent").withRequired(false).withArgument(
      abuilder.withName("maxDFPercent").withMinimum(1).withMaximum(1).create()).withDescription(
      "The max percentage of docs for the DF.  Can be used to remove really high frequency terms."
          + " Expressed as an integer between 0 and 100. Default is 99.").withShortName("x").create();
    
    Option minLLROpt = obuilder.withLongName("minLLR").withRequired(false).withArgument(
      abuilder.withName("minLLR").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional)The minimum Log Likelihood Ratio(Float)  Default is " + LLRReducer.DEFAULT_MIN_LLR)
        .withShortName("ml").create();
    
    Option numReduceTasksOpt = obuilder.withLongName("numReducers").withArgument(
      abuilder.withName("numReducers").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) Number of reduce tasks. Default Value: 1").withShortName("nr").create();
    
    Option powerOpt = obuilder.withLongName("norm").withRequired(false).withArgument(
      abuilder.withName("norm").withMinimum(1).withMaximum(1).create()).withDescription(
      "The norm to use, expressed as either a float or \"INF\" if you want to use the Infinite norm.  "
          + "Must be greater or equal to 0.  The default is not to normalize").withShortName("n").create();
    
    Option logNormalizeOpt = obuilder.withLongName("logNormalize").withRequired(false)
    .withDescription(
      "(Optional) Whether output vectors should be logNormalize. If set true else false")
    .withShortName("lnorm").create();
    
    Option maxNGramSizeOpt = obuilder.withLongName("maxNGramSize").withRequired(false).withArgument(
      abuilder.withName("ngramSize").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "(Optional) The maximum size of ngrams to create"
              + " (2 = bigrams, 3 = trigrams, etc) Default Value:1").withShortName("ng").create();
    
    Option sequentialAccessVectorOpt = obuilder.withLongName("sequentialAccessVector").withRequired(false)
        .withDescription(
          "(Optional) Whether output vectors should be SequentialAccessVectors. If set true else false")
        .withShortName("seq").create();
    
    Option namedVectorOpt = obuilder.withLongName("namedVector").withRequired(false)
    .withDescription(
      "(Optional) Whether output vectors should be NamedVectors. If set true else false")
    .withShortName("nv").create();
    
    Option overwriteOutput = obuilder.withLongName("overwrite").withRequired(false).withDescription(
      "If set, overwrite the output directory").withShortName("ow").create();
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();
    
    Group group = gbuilder.withName("Options").withOption(minSupportOpt).withOption(analyzerNameOpt)
        .withOption(chunkSizeOpt).withOption(outputDirOpt).withOption(inputDirOpt).withOption(minDFOpt)
        .withOption(maxDFPercentOpt).withOption(weightOpt).withOption(powerOpt).withOption(minLLROpt)
        .withOption(numReduceTasksOpt).withOption(maxNGramSizeOpt).withOption(overwriteOutput)
        .withOption(helpOpt).withOption(sequentialAccessVectorOpt).withOption(namedVectorOpt).withOption(logNormalizeOpt)
        .create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      Path inputDir = new Path((String) cmdLine.getValue(inputDirOpt));
      Path outputDir = new Path((String) cmdLine.getValue(outputDirOpt));
      
      int chunkSize = 100;
      if (cmdLine.hasOption(chunkSizeOpt)) {
        chunkSize = Integer.parseInt((String) cmdLine.getValue(chunkSizeOpt));
      }
      int minSupport = 2;
      if (cmdLine.hasOption(minSupportOpt)) {
        String minSupportString = (String) cmdLine.getValue(minSupportOpt);
        minSupport = Integer.parseInt(minSupportString);
      }
      
      int maxNGramSize = 1;
      
      if (cmdLine.hasOption(maxNGramSizeOpt)) {
        try {
          maxNGramSize = Integer.parseInt(cmdLine.getValue(maxNGramSizeOpt).toString());
        } catch (NumberFormatException ex) {
          log.warn("Could not parse ngram size option");
        }
      }
      log.info("Maximum n-gram size is: {}", maxNGramSize);
      
      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.overwriteOutput(outputDir);
      }
      
      float minLLRValue = LLRReducer.DEFAULT_MIN_LLR;
      if (cmdLine.hasOption(minLLROpt)) {
        minLLRValue = Float.parseFloat(cmdLine.getValue(minLLROpt).toString());
      }
      log.info("Minimum LLR value: {}", minLLRValue);
      
      int reduceTasks = 1;
      if (cmdLine.hasOption(numReduceTasksOpt)) {
        reduceTasks = Integer.parseInt(cmdLine.getValue(numReduceTasksOpt).toString());
      }
      log.info("Number of reduce tasks: {}", reduceTasks);
      
      Class<? extends Analyzer> analyzerClass = DefaultAnalyzer.class;
      if (cmdLine.hasOption(analyzerNameOpt)) {
        String className = cmdLine.getValue(analyzerNameOpt).toString();
        analyzerClass = Class.forName(className).asSubclass(Analyzer.class);
        // try instantiating it, b/c there isn't any point in setting it if
        // you can't instantiate it
        analyzerClass.newInstance();
      }
      
      boolean processIdf;
      
      if (cmdLine.hasOption(weightOpt)) {
        String wString = cmdLine.getValue(weightOpt).toString();
        if ("tf".equalsIgnoreCase(wString)) {
          processIdf = false;
        } else if ("tfidf".equalsIgnoreCase(wString)) {
          processIdf = true;
        } else {
          throw new OptionException(weightOpt);
        }
      } else {
        processIdf = true;
      }
      
      int minDf = 1;
      if (cmdLine.hasOption(minDFOpt)) {
        minDf = Integer.parseInt(cmdLine.getValue(minDFOpt).toString());
      }
      int maxDFPercent = 99;
      if (cmdLine.hasOption(maxDFPercentOpt)) {
        maxDFPercent = Integer.parseInt(cmdLine.getValue(maxDFPercentOpt).toString());
      }
      
      float norm = PartialVectorMerger.NO_NORMALIZING;
      if (cmdLine.hasOption(powerOpt)) {
        String power = cmdLine.getValue(powerOpt).toString();
        if ("INF".equals(power)) {
          norm = Float.POSITIVE_INFINITY;
        } else {
          norm = Float.parseFloat(power);
        }
      }
      
      boolean logNormalize = false;
      if (cmdLine.hasOption(logNormalizeOpt)) {
        logNormalize = true;
      }
      
      HadoopUtil.overwriteOutput(outputDir);
      Path tokenizedPath = new Path(outputDir, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
      DocumentProcessor.tokenizeDocuments(inputDir, analyzerClass, tokenizedPath);
      
      boolean sequentialAccessOutput = false;
      if (cmdLine.hasOption(sequentialAccessVectorOpt)) {
        sequentialAccessOutput = true;
      }
      
      boolean namedVectors = false;
      if (cmdLine.hasOption(namedVectorOpt)) {
        namedVectors = true;
      }
      
      Configuration conf = new Configuration();
      if (!processIdf) {
        DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath, outputDir, conf, minSupport, maxNGramSize,
          minLLRValue, norm, logNormalize, reduceTasks, chunkSize, sequentialAccessOutput, namedVectors);
      } else if (processIdf) {
        DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath, outputDir, conf, minSupport, maxNGramSize,
          minLLRValue, -1.0f, false, reduceTasks, chunkSize, sequentialAccessOutput, namedVectors);
      
        TFIDFConverter.processTfIdf(
          new Path(outputDir, DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER),
          outputDir, chunkSize, minDf, maxDFPercent, norm, logNormalize,
          sequentialAccessOutput, namedVectors, reduceTasks);
      }
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }
  
}
