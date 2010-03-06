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

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configured;
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
import org.apache.hadoop.mapred.lib.IdentityMapper;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.utils.vectors.text.DictionaryVectorizer;
import org.apache.mahout.utils.vectors.text.DocumentProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Driver for LLR Collocation discovery mapreduce job */
public class CollocDriver extends Configured implements Tool {
  public static final String DEFAULT_OUTPUT_DIRECTORY = "output";
  public static final String SUBGRAM_OUTPUT_DIRECTORY = "subgrams";
  public static final String NGRAM_OUTPUT_DIRECTORY = "ngrams";
  
  public static final String EMIT_UNIGRAMS = "emit-unigrams";
  public static final boolean DEFAULT_EMIT_UNIGRAMS = false;
  
  public static final int DEFAULT_MAX_NGRAM_SIZE = 2;
  public static final int DEFAULT_PASS1_NUM_REDUCE_TASKS = 1;
  
  private static final Logger log = LoggerFactory.getLogger(CollocDriver.class);

  private CollocDriver() {
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new CollocDriver(), args);
  }
  /**
   * @param args
   */
  public int run(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
      abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
      "The Path for input files.").withShortName("i").create();
    
    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
      abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
      "The Path write output to").withShortName("o").create();
    
    Option maxNGramSizeOpt = obuilder.withLongName("maxNGramSize").withRequired(false).withArgument(
      abuilder.withName("ngramSize").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "(Optional) The maximum size of ngrams to create"
              + " (2 = bigrams, 3 = trigrams, etc) Default Value:2").withShortName("ng").create();
    
    Option minSupportOpt = obuilder.withLongName("minSupport").withRequired(false).withArgument(
      abuilder.withName("minSupport").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) Minimum Support. Default Value: " + CollocReducer.DEFAULT_MIN_SUPPORT).withShortName("s")
        .create();
    
    Option minLLROpt = obuilder.withLongName("minLLR").withRequired(false).withArgument(
      abuilder.withName("minLLR").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional)The minimum Log Likelihood Ratio(Float)  Default is " + LLRReducer.DEFAULT_MIN_LLR)
        .withShortName("ml").create();
    
    Option numReduceTasksOpt = obuilder.withLongName("numReducers").withRequired(false).withArgument(
      abuilder.withName("numReducers").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) Number of reduce tasks. Default Value: " + DEFAULT_PASS1_NUM_REDUCE_TASKS)
        .withShortName("nr").create();
    
    Option preprocessOpt = obuilder.withLongName("preprocess").withRequired(false).withDescription(
      "If set, input is SequenceFile<Text,Text> where the value is the document, "
          + " which will be tokenized using the specified analyzer.").withShortName("p").create();
    
    Option unigramOpt = obuilder.withLongName("unigram").withRequired(false).withDescription(
      "If set, unigrams will be emitted in the final output alongside collocations").withShortName("u")
        .create();
    
    Option overwriteOutput = obuilder.withLongName("overwrite").withRequired(false).withDescription(
      "If set, overwrite the output directory").withShortName("w").create();
    
    Option analyzerNameOpt = obuilder.withLongName("analyzerName").withArgument(
      abuilder.withName("analyzerName").withMinimum(1).withMaximum(1).create()).withDescription(
      "The class name of the analyzer").withShortName("a").create();
    
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();
    
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(
      maxNGramSizeOpt).withOption(overwriteOutput).withOption(minSupportOpt).withOption(minLLROpt)
        .withOption(numReduceTasksOpt).withOption(analyzerNameOpt).withOption(preprocessOpt).withOption(
          unigramOpt).withOption(helpOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return 1;
      }
      
      String input = cmdLine.getValue(inputOpt).toString();
      String output = cmdLine.getValue(outputOpt).toString();
      
      int maxNGramSize = DEFAULT_MAX_NGRAM_SIZE;
      
      if (cmdLine.hasOption(maxNGramSizeOpt)) {
        try {
          maxNGramSize = Integer.parseInt(cmdLine.getValue(maxNGramSizeOpt).toString());
        } catch (NumberFormatException ex) {
          log.warn("Could not parse ngram size option");
        }
      }
      log.info("Maximum n-gram size is: {}", maxNGramSize);
      
      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.overwriteOutput(output);
      }
      
      int minSupport = CollocReducer.DEFAULT_MIN_SUPPORT;
      if (cmdLine.hasOption(minSupportOpt)) {
        minSupport = Integer.parseInt(cmdLine.getValue(minSupportOpt).toString());
      }
      log.info("Minimum Support value: {}", minSupport);
      
      float minLLRValue = LLRReducer.DEFAULT_MIN_LLR;
      if (cmdLine.hasOption(minLLROpt)) {
        minLLRValue = Float.parseFloat(cmdLine.getValue(minLLROpt).toString());
      }
      log.info("Minimum LLR value: {}", minLLRValue);
      
      int reduceTasks = DEFAULT_PASS1_NUM_REDUCE_TASKS;
      if (cmdLine.hasOption(numReduceTasksOpt)) {
        reduceTasks = Integer.parseInt(cmdLine.getValue(numReduceTasksOpt).toString());
      }
      log.info("Number of pass1 reduce tasks: {}", reduceTasks);
      
      boolean emitUnigrams = cmdLine.hasOption(unigramOpt);
      
      if (cmdLine.hasOption(preprocessOpt)) {
        log.info("Input will be preprocessed");
        
        Class<? extends Analyzer> analyzerClass = StandardAnalyzer.class;
        if (cmdLine.hasOption(analyzerNameOpt)) {
          String className = cmdLine.getValue(analyzerNameOpt).toString();
          analyzerClass = Class.forName(className).asSubclass(Analyzer.class);
          // try instantiating it, b/c there isn't any point in setting it if
          // you can't instantiate it
          analyzerClass.newInstance();
        }
        
        String tokenizedPath = output + DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER;
        
        DocumentProcessor.tokenizeDocuments(input, analyzerClass, tokenizedPath);
        input = tokenizedPath;
      } else {
        log.info("Input will NOT be preprocessed");
      }
      
      // parse input and extract collocations
      long ngramCount = generateCollocations(input, output, emitUnigrams, maxNGramSize,
        reduceTasks, minSupport);
      
      // tally collocations and perform LLR calculation
      computeNGramsPruneByLLR(ngramCount, output, emitUnigrams, minLLRValue, reduceTasks);
      
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
      return 1;
    }
    
    return 0;
  }
  
  /**
   * Generate all ngrams for the {@link DictionaryVectorizer} job
   * 
   * @param input
   *          input path containing tokenized documents
   * @param output
   *          output path where ngrams are generated including unigrams
   * @param maxNGramSize
   *          minValue = 2.
   * @param minSupport
   *          minimum support to prune ngrams including unigrams
   * @param minLLRValue
   *          minimum threshold to prune ngrams
   * @param reduceTasks
   *          number of reducers used
   * @throws IOException
   */
  public static void generateAllGrams(String input,
                                      String output,
                                      int maxNGramSize,
                                      int minSupport,
                                      float minLLRValue,
                                      int reduceTasks) throws IOException {
    // parse input and extract collocations
    long ngramCount = generateCollocations(input, output, true, maxNGramSize, reduceTasks,
      minSupport);
    
    // tally collocations and perform LLR calculation
    computeNGramsPruneByLLR(ngramCount, output, true, minLLRValue, reduceTasks);
  }
  
  /**
   * pass1: generate collocations, ngrams
   */
  public static long generateCollocations(String input,
                                          String output,
                                          boolean emitUnigrams,
                                          int maxNGramSize,
                                          int reduceTasks,
                                          int minSupport) throws IOException {
    JobConf conf = new JobConf(CollocDriver.class);
    conf.setJobName(CollocDriver.class.getSimpleName() + ".generateCollocations:" + input);
    
    conf.setMapOutputKeyClass(GramKey.class);
    conf.setMapOutputValueClass(Gram.class);
    conf.setPartitionerClass(GramKeyPartitioner.class);
    conf.setOutputValueGroupingComparator(GramKeyGroupComparator.class);
    
    conf.setOutputKeyClass(Gram.class);
    conf.setOutputValueClass(Gram.class);
    
    conf.setCombinerClass(CollocCombiner.class);
    
    conf.setBoolean(EMIT_UNIGRAMS, emitUnigrams);
    
    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output, SUBGRAM_OUTPUT_DIRECTORY);
    FileOutputFormat.setOutputPath(conf, outPath);
    
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setMapperClass(CollocMapper.class);
    
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.setReducerClass(CollocReducer.class);
    conf.setInt(CollocMapper.MAX_SHINGLE_SIZE, maxNGramSize);
    conf.setInt(CollocReducer.MIN_SUPPORT, minSupport);
    conf.setNumReduceTasks(reduceTasks);
    
    RunningJob job = JobClient.runJob(conf);
    return job.getCounters().findCounter(CollocMapper.Count.NGRAM_TOTAL).getValue();
  }
  
  /**
   * pass2: perform the LLR calculation
   */
  public static void computeNGramsPruneByLLR(long nGramTotal,
                                             String output,
                                             boolean emitUnigrams,
                                             float minLLRValue,
                                             int reduceTasks) throws IOException {
    JobConf conf = new JobConf(CollocDriver.class);
    conf.setJobName(CollocDriver.class.getSimpleName() + ".computeNGrams: " + output);
    
    
    conf.setLong(LLRReducer.NGRAM_TOTAL, nGramTotal);
    conf.setBoolean(EMIT_UNIGRAMS, emitUnigrams);
    
    conf.setMapOutputKeyClass(Gram.class);
    conf.setMapOutputValueClass(Gram.class);
    
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(DoubleWritable.class);
    
    FileInputFormat.setInputPaths(conf, new Path(output, SUBGRAM_OUTPUT_DIRECTORY));
    Path outPath = new Path(output, NGRAM_OUTPUT_DIRECTORY);
    FileOutputFormat.setOutputPath(conf, outPath);
    
    conf.setMapperClass(IdentityMapper.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.setReducerClass(LLRReducer.class);
    conf.setNumReduceTasks(reduceTasks);
    
    conf.setFloat(LLRReducer.MIN_LLR, minLLRValue);
    JobClient.runJob(conf);
  }
}
