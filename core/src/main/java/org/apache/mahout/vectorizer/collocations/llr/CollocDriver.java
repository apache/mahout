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

package org.apache.mahout.vectorizer.collocations.llr;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.lucene.AnalyzerUtils;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Driver for LLR Collocation discovery mapreduce job */
public final class CollocDriver extends AbstractJob {
  //public static final String DEFAULT_OUTPUT_DIRECTORY = "output";

  public static final String SUBGRAM_OUTPUT_DIRECTORY = "subgrams";

  public static final String NGRAM_OUTPUT_DIRECTORY = "ngrams";

  public static final String EMIT_UNIGRAMS = "emit-unigrams";

  public static final boolean DEFAULT_EMIT_UNIGRAMS = false;

  private static final int DEFAULT_MAX_NGRAM_SIZE = 2;

  private static final int DEFAULT_PASS1_NUM_REDUCE_TASKS = 1;

  private static final Logger log = LoggerFactory.getLogger(CollocDriver.class);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new CollocDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.numReducersOption().create());

    addOption("maxNGramSize",
              "ng",
              "(Optional) The max size of ngrams to create (2 = bigrams, 3 = trigrams, etc) default: 2",
              String.valueOf(DEFAULT_MAX_NGRAM_SIZE));
    addOption("minSupport", "s", "(Optional) Minimum Support. Default Value: "
        + CollocReducer.DEFAULT_MIN_SUPPORT, String.valueOf(CollocReducer.DEFAULT_MIN_SUPPORT));
    addOption("minLLR", "ml", "(Optional)The minimum Log Likelihood Ratio(Float)  Default is "
        + LLRReducer.DEFAULT_MIN_LLR, String.valueOf(LLRReducer.DEFAULT_MIN_LLR));
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption("analyzerName", "a", "The class name of the analyzer to use for preprocessing", null);

    addFlag("preprocess", "p", "If set, input is SequenceFile<Text,Text> where the value is the document, "
        + " which will be tokenized using the specified analyzer.");
    addFlag("unigram", "u", "If set, unigrams will be emitted in the final output alongside collocations");

    Map<String, List<String>> argMap = parseArguments(args);

    if (argMap == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();

    int maxNGramSize = DEFAULT_MAX_NGRAM_SIZE;
    if (hasOption("maxNGramSize")) {
      try {
        maxNGramSize = Integer.parseInt(getOption("maxNGramSize"));
      } catch (NumberFormatException ex) {
        log.warn("Could not parse ngram size option");
      }
    }
    log.info("Maximum n-gram size is: {}", maxNGramSize);

    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }

    int minSupport = CollocReducer.DEFAULT_MIN_SUPPORT;
    if (getOption("minSupport") != null) {
      minSupport = Integer.parseInt(getOption("minSupport"));
    }
    log.info("Minimum Support value: {}", minSupport);

    float minLLRValue = LLRReducer.DEFAULT_MIN_LLR;
    if (getOption("minLLR") != null) {
      minLLRValue = Float.parseFloat(getOption("minLLR"));
    }
    log.info("Minimum LLR value: {}", minLLRValue);

    int reduceTasks = DEFAULT_PASS1_NUM_REDUCE_TASKS;
    if (getOption("maxRed") != null) {
      reduceTasks = Integer.parseInt(getOption("maxRed"));
    }
    log.info("Number of pass1 reduce tasks: {}", reduceTasks);

    boolean emitUnigrams = argMap.containsKey("emitUnigrams");

    if (argMap.containsKey("preprocess")) {
      log.info("Input will be preprocessed");
      Class<? extends Analyzer> analyzerClass = StandardAnalyzer.class;
      if (getOption("analyzerName") != null) {
        String className = getOption("analyzerName");
        analyzerClass = Class.forName(className).asSubclass(Analyzer.class);
        // try instantiating it, b/c there isn't any point in setting it if
        // you can't instantiate it
        AnalyzerUtils.createAnalyzer(analyzerClass);
      }

      Path tokenizedPath = new Path(output, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);

      DocumentProcessor.tokenizeDocuments(input, analyzerClass, tokenizedPath, getConf());
      input = tokenizedPath;
    } else {
      log.info("Input will NOT be preprocessed");
    }

    // parse input and extract collocations
    long ngramCount =
      generateCollocations(input, output, getConf(), emitUnigrams, maxNGramSize, reduceTasks, minSupport);

    // tally collocations and perform LLR calculation
    computeNGramsPruneByLLR(output, getConf(), ngramCount, emitUnigrams, minLLRValue, reduceTasks);

    return 0;
  }

  /**
   * Generate all ngrams for the {@link org.apache.mahout.vectorizer.DictionaryVectorizer} job
   * 
   * @param input
   *          input path containing tokenized documents
   * @param output
   *          output path where ngrams are generated including unigrams
   * @param baseConf
   *          job configuration
   * @param maxNGramSize
   *          minValue = 2.
   * @param minSupport
   *          minimum support to prune ngrams including unigrams
   * @param minLLRValue
   *          minimum threshold to prune ngrams
   * @param reduceTasks
   *          number of reducers used
   */
  public static void generateAllGrams(Path input,
                                      Path output,
                                      Configuration baseConf,
                                      int maxNGramSize,
                                      int minSupport,
                                      float minLLRValue,
                                      int reduceTasks)
    throws IOException, InterruptedException, ClassNotFoundException {
    // parse input and extract collocations
    long ngramCount = generateCollocations(input, output, baseConf, true, maxNGramSize, reduceTasks, minSupport);

    // tally collocations and perform LLR calculation
    computeNGramsPruneByLLR(output, baseConf, ngramCount, true, minLLRValue, reduceTasks);
  }

  /**
   * pass1: generate collocations, ngrams
   */
  private static long generateCollocations(Path input,
                                           Path output,
                                           Configuration baseConf,
                                           boolean emitUnigrams,
                                           int maxNGramSize,
                                           int reduceTasks,
                                           int minSupport)
    throws IOException, ClassNotFoundException, InterruptedException {

    Configuration con = new Configuration(baseConf);
    con.setBoolean(EMIT_UNIGRAMS, emitUnigrams);
    con.setInt(CollocMapper.MAX_SHINGLE_SIZE, maxNGramSize);
    con.setInt(CollocReducer.MIN_SUPPORT, minSupport);
    
    Job job = new Job(con);
    job.setJobName(CollocDriver.class.getSimpleName() + ".generateCollocations:" + input);
    job.setJarByClass(CollocDriver.class);
    
    job.setMapOutputKeyClass(GramKey.class);
    job.setMapOutputValueClass(Gram.class);
    job.setPartitionerClass(GramKeyPartitioner.class);
    job.setGroupingComparatorClass(GramKeyGroupComparator.class);

    job.setOutputKeyClass(Gram.class);
    job.setOutputValueClass(Gram.class);

    job.setCombinerClass(CollocCombiner.class);

    FileInputFormat.setInputPaths(job, input);

    Path outputPath = new Path(output, SUBGRAM_OUTPUT_DIRECTORY);
    FileOutputFormat.setOutputPath(job, outputPath);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapperClass(CollocMapper.class);

    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setReducerClass(CollocReducer.class);
    job.setNumReduceTasks(reduceTasks);
    
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }

    return job.getCounters().findCounter(CollocMapper.Count.NGRAM_TOTAL).getValue();
  }

  /**
   * pass2: perform the LLR calculation
   */
  private static void computeNGramsPruneByLLR(Path output,
                                              Configuration baseConf,
                                              long nGramTotal,
                                              boolean emitUnigrams,
                                              float minLLRValue,
                                              int reduceTasks)
    throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration(baseConf);
    conf.setLong(LLRReducer.NGRAM_TOTAL, nGramTotal);
    conf.setBoolean(EMIT_UNIGRAMS, emitUnigrams);
    conf.setFloat(LLRReducer.MIN_LLR, minLLRValue);

    Job job = new Job(conf);
    job.setJobName(CollocDriver.class.getSimpleName() + ".computeNGrams: " + output);
    job.setJarByClass(CollocDriver.class);
    
    job.setMapOutputKeyClass(Gram.class);
    job.setMapOutputValueClass(Gram.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(DoubleWritable.class);

    FileInputFormat.setInputPaths(job, new Path(output, SUBGRAM_OUTPUT_DIRECTORY));
    Path outPath = new Path(output, NGRAM_OUTPUT_DIRECTORY);
    FileOutputFormat.setOutputPath(job, outPath);

    job.setMapperClass(Mapper.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setReducerClass(LLRReducer.class);
    job.setNumReduceTasks(reduceTasks);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }
}
