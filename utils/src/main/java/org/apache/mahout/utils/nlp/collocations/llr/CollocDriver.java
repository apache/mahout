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

package org.apache.mahout.utils.nlp.collocations.llr;

import static org.apache.mahout.utils.nlp.collocations.llr.NGramCollector.Count.NGRAM_TOTAL;

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityMapper;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Driver for LLR collocation discovery mapreduce job */
public class CollocDriver {
  
  public static final String DEFAULT_OUTPUT_DIRECTORY = "output";
  public static final int DEFAULT_MAX_NGRAM_SIZE = 2;
  
  private static final Logger log = LoggerFactory.getLogger(CollocDriver.class);
  
  /**
   * @param args
   */
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = obuilder.withLongName("input").withRequired(true)
        .withArgument(
          abuilder.withName("input").withMinimum(1).withMaximum(1).create())
        .withDescription("The Path for input files.").withShortName("i")
        .create();
    
    Option outputOpt = obuilder.withLongName("output").withRequired(true)
        .withArgument(
          abuilder.withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription("The Path write output to").withShortName("o")
        .create();
    
    Option maxNGramSizeOpt = obuilder.withLongName("maxNGramSize")
        .withRequired(false).withArgument(
          abuilder.withName("size").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "(Optional) The maximum size of ngrams to create"
              + " (2 = bigrams, 3 = trigrams, etc) Default Value:2")
        .withShortName("n").create();
    
    Option minSupportOpt = obuilder.withLongName("minSupport").withArgument(
      abuilder.withName("minSupport").withMinimum(1).withMaximum(1).create())
        .withDescription("(Optional) Minimum Support. Default Value: 2")
        .withShortName("s").create();
    
    Option minLLROpt = obuilder
        .withLongName("minLLR")
        .withRequired(false)
        .withArgument(
          abuilder.withName("minDF").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "(Optional)The minimum Log Likelihood Ratio(Float)  Default is 0.00")
        .withShortName("ml").create();
    
    Option overwriteOutput = obuilder.withLongName("overwrite").withRequired(
      false).withDescription("If set, overwrite the output directory")
        .withShortName("w").create();
    
    Option analyzerNameOpt = obuilder.withLongName("analyzerName")
        .withRequired(false)
        .withArgument(
          abuilder.withName("analyzerName").withMinimum(1).withMaximum(1)
              .create())
        .withDescription(
          "Class name of analyzer to use for tokenization").withShortName("a")
        .create();
    
    Option helpOpt = obuilder.withLongName("help").withDescription(
      "Print out help").withShortName("h").create();
    
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(
      outputOpt).withOption(maxNGramSizeOpt).withOption(overwriteOutput)
        .withOption(analyzerNameOpt).withOption(minSupportOpt).withOption(
          minLLROpt).withOption(helpOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      String input = cmdLine.getValue(inputOpt).toString();
      String output = cmdLine.getValue(outputOpt).toString();
      
      int maxNGramSize = DEFAULT_MAX_NGRAM_SIZE;
      
      if (cmdLine.hasOption(maxNGramSizeOpt) == true) {
        try {
          maxNGramSize = Integer.parseInt(cmdLine.getValue(maxNGramSizeOpt)
              .toString());
        } catch (NumberFormatException ex) {
          log.warn("Could not parse ngram size option");
        }
      }
      
      if (cmdLine.hasOption(overwriteOutput) == true) {
        HadoopUtil.overwriteOutput(output);
      }
      
      String analyzerName = null;
      if (cmdLine.hasOption(analyzerNameOpt) == true) {
        analyzerName = cmdLine.getValue(analyzerNameOpt).toString();
      }
      
      int minSupport = 2;
      if (cmdLine.hasOption(minSupportOpt)) {
        minSupport = Integer.parseInt(cmdLine.getValue(minSupportOpt)
            .toString());
      }
      
      float minLLRValue = 1.0f;
      if (cmdLine.hasOption(minLLROpt)) {
        minLLRValue = Float
            .parseFloat(cmdLine.getValue(minLLROpt).toString());
      }
      
      // parse input and extract collocations
      long ngramCount = runPass1(input, output, maxNGramSize, analyzerName,
        minSupport);
      
      // tally collocations and perform LLR calculation
      runPass2(ngramCount, output, minLLRValue);
      
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
    
  }
  
  /**
   * pass1: generate collocations, ngrams
   */
  public static long runPass1(String input,
                              String output,
                              int maxNGramSize,
                              String analyzerClass,
                              int minSupport) throws IOException {
    JobConf conf = new JobConf(CollocDriver.class);
    
    conf.setMapOutputKeyClass(Gram.class);
    conf.setMapOutputValueClass(Gram.class);
    
    conf.setOutputKeyClass(Gram.class);
    conf.setOutputValueClass(Gram.class);
    
    conf.setCombinerClass(CollocCombiner.class);
    
    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output + "/pass1");
    FileOutputFormat.setOutputPath(conf, outPath);
    
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setMapperClass(CollocMapper.class);
    
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.setReducerClass(CollocReducer.class);
    conf.setInt(NGramCollector.MAX_SHINGLE_SIZE, maxNGramSize);
    conf.setInt(CollocReducer.MIN_SUPPORT, minSupport);
    
    if (analyzerClass != null) {
      conf.set(NGramCollector.ANALYZER_CLASS, analyzerClass);
    }
    
    RunningJob job = JobClient.runJob(conf);
    return job.getCounters().findCounter(NGRAM_TOTAL).getValue();
  }
  
  /**
   * pass2: perform the LLR calculation
   */
  public static void runPass2(long nGramTotal,
                              String output,
                              float minLLRValue) throws IOException {
    JobConf conf = new JobConf(CollocDriver.class);
    conf.set(LLRReducer.NGRAM_TOTAL, String.valueOf(nGramTotal));
    
    conf.setMapOutputKeyClass(Gram.class);
    conf.setMapOutputValueClass(Gram.class);
    
    conf.setOutputKeyClass(DoubleWritable.class);
    conf.setOutputValueClass(Text.class);
    
    FileInputFormat.setInputPaths(conf, new Path(output + "/pass1"));
    Path outPath = new Path(output + "/colloc");
    FileOutputFormat.setOutputPath(conf, outPath);
    
    conf.setMapperClass(IdentityMapper.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(TextOutputFormat.class);
    conf.setReducerClass(LLRReducer.class);
    
    conf.setFloat(LLRReducer.MIN_LLR, minLLRValue);
    JobClient.runJob(conf);
  }
}
