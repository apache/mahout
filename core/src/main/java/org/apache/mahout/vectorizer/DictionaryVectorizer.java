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

import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.List;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.collocations.llr.CollocDriver;
import org.apache.mahout.vectorizer.collocations.llr.LLRReducer;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.term.TFPartialVectorReducer;
import org.apache.mahout.vectorizer.term.TermCountCombiner;
import org.apache.mahout.vectorizer.term.TermCountMapper;
import org.apache.mahout.vectorizer.term.TermCountReducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class converts a set of input documents in the sequence file format to vectors. The Sequence file
 * input should have a {@link Text} key containing the unique document identifier and a {@link StringTuple}
 * value containing the tokenized document. You may use {@link DocumentProcessor} to tokenize the document.
 * This is a dictionary based Vectorizer.
 */
public final class DictionaryVectorizer extends AbstractJob implements Vectorizer {
  private static final Logger log = LoggerFactory.getLogger(DictionaryVectorizer.class);
  
  public static final String DOCUMENT_VECTOR_OUTPUT_FOLDER = "tf-vectors";
  public static final String MIN_SUPPORT = "min.support";
  public static final String MAX_NGRAMS = "max.ngrams";
  public static final int DEFAULT_MIN_SUPPORT = 2;
  
  private static final String DICTIONARY_FILE = "dictionary.file-";
  private static final int MAX_CHUNKSIZE = 10000;
  private static final int MIN_CHUNKSIZE = 100;
  private static final String OUTPUT_FILES_PATTERN = "part-*";
  // 4 byte overhead for each entry in the OpenObjectIntHashMap
  private static final int DICTIONARY_BYTE_OVERHEAD = 4;
  private static final String VECTOR_OUTPUT_FOLDER = "partial-vectors-";
  private static final String DICTIONARY_JOB_FOLDER = "wordcount";
  
  /**
   * Cannot be initialized. Use the static functions
   */
  private DictionaryVectorizer() {
  }

  //TODO: move more of SparseVectorsFromSequenceFile in here, and then fold SparseVectorsFrom with
  // EncodedVectorsFrom to have one framework.

  @Override
  public void createVectors(Path input, Path output, VectorizerConfig config)
    throws IOException, ClassNotFoundException, InterruptedException {
    createTermFrequencyVectors(input,
                               output,
                               config.getTfDirName(),
                               config.getConf(),
                               config.getMinSupport(),
                               config.getMaxNGramSize(),
                               config.getMinLLRValue(),
                               config.getNormPower(),
                               config.isLogNormalize(),
                               config.getNumReducers(),
                               config.getChunkSizeInMegabytes(),
                               config.isSequentialAccess(),
                               config.isNamedVectors());
  }

  /**
   * Create Term Frequency (Tf) Vectors from the input set of documents in {@link SequenceFile} format. This
   * tries to fix the maximum memory used by the feature chunk per node thereby splitting the process across
   * multiple map/reduces.
   * 
   * @param input
   *          input directory of the documents in {@link SequenceFile} format
   * @param output
   *          output directory where {@link org.apache.mahout.math.RandomAccessSparseVector}'s of the document
   *          are generated
   * @param tfVectorsFolderName
   *          The name of the folder in which the final output vectors will be stored
   * @param baseConf
   *          job configuration
   * @param normPower
   *          L_p norm to be computed
   * @param logNormalize
   *          whether to use log normalization         
   * @param minSupport
   *          the minimum frequency of the feature in the entire corpus to be considered for inclusion in the
   *          sparse vector
   * @param maxNGramSize
   *          1 = unigram, 2 = unigram and bigram, 3 = unigram, bigram and trigram
   * @param minLLRValue
   *          minValue of log likelihood ratio to used to prune ngrams
   * @param chunkSizeInMegabytes
   *          the size in MB of the feature => id chunk to be kept in memory at each node during Map/Reduce
   *          stage. Its recommended you calculated this based on the number of cores and the free memory
   *          available to you per node. Say, you have 2 cores and around 1GB extra memory to spare we
   *          recommend you use a split size of around 400-500MB so that two simultaneous reducers can create
   *          partial vectors without thrashing the system due to increased swapping
   */
  public static void createTermFrequencyVectors(Path input,
                                                Path output,
                                                String tfVectorsFolderName,
                                                Configuration baseConf,
                                                int minSupport,
                                                int maxNGramSize,
                                                float minLLRValue,
                                                float normPower,
                                                boolean logNormalize,
                                                int numReducers,
                                                int chunkSizeInMegabytes,
                                                boolean sequentialAccess,
                                                boolean namedVectors)
    throws IOException, InterruptedException, ClassNotFoundException {
    Preconditions.checkArgument(normPower == PartialVectorMerger.NO_NORMALIZING || normPower >= 0,
        "If specified normPower must be nonnegative", normPower);
    Preconditions.checkArgument(normPower == PartialVectorMerger.NO_NORMALIZING 
                                || (normPower > 1 && !Double.isInfinite(normPower))
                                || !logNormalize,
        "normPower must be > 1 and not infinite if log normalization is chosen", normPower);
    if (chunkSizeInMegabytes < MIN_CHUNKSIZE) {
      chunkSizeInMegabytes = MIN_CHUNKSIZE;
    } else if (chunkSizeInMegabytes > MAX_CHUNKSIZE) { // 10GB
      chunkSizeInMegabytes = MAX_CHUNKSIZE;
    }
    if (minSupport < 0) {
      minSupport = DEFAULT_MIN_SUPPORT;
    }
    
    Path dictionaryJobPath = new Path(output, DICTIONARY_JOB_FOLDER);
    
    log.info("Creating dictionary from {} and saving at {}", input, dictionaryJobPath);
    
    int[] maxTermDimension = new int[1];
    List<Path> dictionaryChunks;
    if (maxNGramSize == 1) {
      startWordCounting(input, dictionaryJobPath, baseConf, minSupport);
      dictionaryChunks =
          createDictionaryChunks(dictionaryJobPath, output, baseConf, chunkSizeInMegabytes, maxTermDimension);
    } else {
      CollocDriver.generateAllGrams(input, dictionaryJobPath, baseConf, maxNGramSize,
        minSupport, minLLRValue, numReducers);
      dictionaryChunks =
          createDictionaryChunks(new Path(new Path(output, DICTIONARY_JOB_FOLDER),
                                          CollocDriver.NGRAM_OUTPUT_DIRECTORY),
                                 output,
                                 baseConf,
                                 chunkSizeInMegabytes,
                                 maxTermDimension);
    }
    
    int partialVectorIndex = 0;
    Collection<Path> partialVectorPaths = Lists.newArrayList();
    for (Path dictionaryChunk : dictionaryChunks) {
      Path partialVectorOutputPath = new Path(output, VECTOR_OUTPUT_FOLDER + partialVectorIndex++);
      partialVectorPaths.add(partialVectorOutputPath);
      makePartialVectors(input, baseConf, maxNGramSize, dictionaryChunk, partialVectorOutputPath,
        maxTermDimension[0], sequentialAccess, namedVectors, numReducers);
    }
    
    Configuration conf = new Configuration(baseConf);

    Path outputDir = new Path(output, tfVectorsFolderName);
    PartialVectorMerger.mergePartialVectors(partialVectorPaths, outputDir, conf, normPower, logNormalize,
      maxTermDimension[0], sequentialAccess, namedVectors, numReducers);
    HadoopUtil.delete(conf, partialVectorPaths);
  }
  
  /**
   * Read the feature frequency List which is built at the end of the Word Count Job and assign ids to them.
   * This will use constant memory and will run at the speed of your disk read
   */
  private static List<Path> createDictionaryChunks(Path wordCountPath,
                                                   Path dictionaryPathBase,
                                                   Configuration baseConf,
                                                   int chunkSizeInMegabytes,
                                                   int[] maxTermDimension) throws IOException {
    List<Path> chunkPaths = Lists.newArrayList();
    
    Configuration conf = new Configuration(baseConf);
    
    FileSystem fs = FileSystem.get(wordCountPath.toUri(), conf);

    long chunkSizeLimit = chunkSizeInMegabytes * 1024L * 1024L;
    int chunkIndex = 0;
    Path chunkPath = new Path(dictionaryPathBase, DICTIONARY_FILE + chunkIndex);
    chunkPaths.add(chunkPath);
    
    SequenceFile.Writer dictWriter = new SequenceFile.Writer(fs, conf, chunkPath, Text.class, IntWritable.class);

    try {
      long currentChunkSize = 0;
      Path filesPattern = new Path(wordCountPath, OUTPUT_FILES_PATTERN);
      int i = 0;
      for (Pair<Writable,Writable> record
           : new SequenceFileDirIterable<Writable,Writable>(filesPattern, PathType.GLOB, null, null, true, conf)) {
        if (currentChunkSize > chunkSizeLimit) {
          Closeables.close(dictWriter, false);
          chunkIndex++;

          chunkPath = new Path(dictionaryPathBase, DICTIONARY_FILE + chunkIndex);
          chunkPaths.add(chunkPath);

          dictWriter = new SequenceFile.Writer(fs, conf, chunkPath, Text.class, IntWritable.class);
          currentChunkSize = 0;
        }

        Writable key = record.getFirst();
        int fieldSize = DICTIONARY_BYTE_OVERHEAD + key.toString().length() * 2 + Integer.SIZE / 8;
        currentChunkSize += fieldSize;
        dictWriter.append(key, new IntWritable(i++));
      }
      maxTermDimension[0] = i;
    } finally {
      Closeables.close(dictWriter, false);
    }
    
    return chunkPaths;
  }
  
  /**
   * Create a partial vector using a chunk of features from the input documents. The input documents has to be
   * in the {@link SequenceFile} format
   * 
   * @param input
   *          input directory of the documents in {@link SequenceFile} format
   * @param baseConf
   *          job configuration
   * @param maxNGramSize
   *          maximum size of ngrams to generate
   * @param dictionaryFilePath
   *          location of the chunk of features and the id's
   * @param output
   *          output directory were the partial vectors have to be created
   * @param dimension
   * @param sequentialAccess
   *          output vectors should be optimized for sequential access
   * @param namedVectors
   *          output vectors should be named, retaining key (doc id) as a label
   * @param numReducers 
   *          the desired number of reducer tasks
   */
  private static void makePartialVectors(Path input,
                                         Configuration baseConf,
                                         int maxNGramSize,
                                         Path dictionaryFilePath,
                                         Path output,
                                         int dimension,
                                         boolean sequentialAccess, 
                                         boolean namedVectors,
                                         int numReducers)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    Configuration conf = new Configuration(baseConf);
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
                                  + "org.apache.hadoop.io.serializer.WritableSerialization");
    conf.setInt(PartialVectorMerger.DIMENSION, dimension);
    conf.setBoolean(PartialVectorMerger.SEQUENTIAL_ACCESS, sequentialAccess);
    conf.setBoolean(PartialVectorMerger.NAMED_VECTOR, namedVectors);
    conf.setInt(MAX_NGRAMS, maxNGramSize);   
    DistributedCache.setCacheFiles(new URI[] {dictionaryFilePath.toUri()}, conf);
    
    Job job = new Job(conf);
    job.setJobName("DictionaryVectorizer::MakePartialVectors: input-folder: " + input
                    + ", dictionary-file: " + dictionaryFilePath);
    job.setJarByClass(DictionaryVectorizer.class);
    
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(StringTuple.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(VectorWritable.class);
    FileInputFormat.setInputPaths(job, input);
    
    FileOutputFormat.setOutputPath(job, output);
    
    job.setMapperClass(Mapper.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setReducerClass(TFPartialVectorReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setNumReduceTasks(numReducers);

    HadoopUtil.delete(conf, output);
    
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }
  
  /**
   * Count the frequencies of words in parallel using Map/Reduce. The input documents have to be in
   * {@link SequenceFile} format
   */
  private static void startWordCounting(Path input, Path output, Configuration baseConf, int minSupport)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    Configuration conf = new Configuration(baseConf);
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
                                  + "org.apache.hadoop.io.serializer.WritableSerialization");
    conf.setInt(MIN_SUPPORT, minSupport);
    
    Job job = new Job(conf);
    
    job.setJobName("DictionaryVectorizer::WordCount: input-folder: " + input);
    job.setJarByClass(DictionaryVectorizer.class);
    
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(LongWritable.class);
    
    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);
    
    job.setMapperClass(TermCountMapper.class);
    
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setCombinerClass(TermCountCombiner.class);
    job.setReducerClass(TermCountReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    HadoopUtil.delete(conf, output);
    
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption("tfDirName", "tf", "The folder to store the TF calculations", "tfDirName");
    addOption("minSupport", "s", "(Optional) Minimum Support. Default Value: 2", "2");
    addOption("maxNGramSize", "ng", "(Optional) The maximum size of ngrams to create"
                            + " (2 = bigrams, 3 = trigrams, etc) Default Value:1");
    addOption("minLLR", "ml", "(Optional)The minimum Log Likelihood Ratio(Float)  Default is "
        + LLRReducer.DEFAULT_MIN_LLR);
    addOption("norm", "n", "The norm to use, expressed as either a float or \"INF\" "
        + "if you want to use the Infinite norm.  "
                    + "Must be greater or equal to 0.  The default is not to normalize");
    addOption("logNormalize", "lnorm", "(Optional) Whether output vectors should be logNormalize. "
        + "If set true else false", "false");
    addOption(DefaultOptionCreator.numReducersOption().create());
    addOption("chunkSize", "chunk", "The chunkSize in MegaBytes. 100-10000 MB", "100");
    addOption(DefaultOptionCreator.methodOption().create());
    addOption("namedVector", "nv", "(Optional) Whether output vectors should be NamedVectors. "
        + "If set true else false", "false");
    if (parseArguments(args) == null) {
      return -1;
    }
    String tfDirName = getOption("tfDirName", "tfDir");
    int minSupport = getInt("minSupport", 2);
    int maxNGramSize = getInt("maxNGramSize", 1);
    float minLLRValue = getFloat("minLLR", LLRReducer.DEFAULT_MIN_LLR);
    float normPower = getFloat("norm", PartialVectorMerger.NO_NORMALIZING);
    boolean logNormalize = hasOption("logNormalize");
    int numReducers = getInt(DefaultOptionCreator.MAX_REDUCERS_OPTION);
    int chunkSizeInMegs = getInt("chunkSize", 100);
    boolean sequential = hasOption("sequential");
    boolean namedVecs = hasOption("namedVectors");
    //TODO: add support for other paths
    createTermFrequencyVectors(getInputPath(), getOutputPath(),
            tfDirName,
            getConf(), minSupport, maxNGramSize, minLLRValue,
            normPower, logNormalize, numReducers, chunkSizeInMegs, sequential, namedVecs);
    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DictionaryVectorizer(), args);
  }
}
