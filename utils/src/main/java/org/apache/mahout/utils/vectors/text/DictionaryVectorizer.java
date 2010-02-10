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

package org.apache.mahout.utils.vectors.text;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityMapper;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.nlp.collocations.llr.CollocDriver;
import org.apache.mahout.utils.vectors.common.PartialVectorMerger;
import org.apache.mahout.utils.vectors.text.term.TFPartialVectorReducer;
import org.apache.mahout.utils.vectors.text.term.TermCountMapper;
import org.apache.mahout.utils.vectors.text.term.TermCountReducer;

/**
 * This class converts a set of input documents in the sequence file format to
 * vectors. The Sequence file input should have a {@link Text} key containing
 * the unique document identifier and a {@link StringTuple} value containing the
 * tokenized document. You may use {@link DocumentProcessor} to tokenize the
 * document. This is a dictionary based Vectorizer.
 * 
 */
public final class DictionaryVectorizer {
  
  public static final String DOCUMENT_VECTOR_OUTPUT_FOLDER = "/vectors";
  
  public static final String MIN_SUPPORT = "min.support";
  
  public static final String MAX_NGRAMS = "max.ngrams";
  
  public static final int DEFAULT_MIN_SUPPORT = 2;
  
  private static final String DICTIONARY_FILE = "/dictionary.file-";
  
  private static final int MAX_CHUNKSIZE = 10000;
  
  private static final int MIN_CHUNKSIZE = 100;
  
  private static final String OUTPUT_FILES_PATTERN = "/part-*";
  
  // 4 byte overhead for each entry in the OpenObjectIntHashMap
  private static final int DICTIONARY_BYTE_OVERHEAD = 4;
  
  private static final String VECTOR_OUTPUT_FOLDER = "/partial-vectors-";
  
  private static final String DICTIONARY_JOB_FOLDER = "/wordcount";
  
  /**
   * Cannot be initialized. Use the static functions
   */
  private DictionaryVectorizer() {

  }
  
  /**
   * Create Term Frequency (Tf) Vectors from the input set of documents in
   * {@link SequenceFile} format. This tries to fix the maximum memory used by
   * the feature chunk per node thereby splitting the process across multiple
   * map/reduces.
   * 
   * @param input
   *          input directory of the documents in {@link SequenceFile} format
   * @param output
   *          output directory where
   *          {@link org.apache.mahout.math.RandomAccessSparseVector}'s of the
   *          document are generated
   * @param minSupport
   *          the minimum frequency of the feature in the entire corpus to be
   *          considered for inclusion in the sparse vector
   * @param maxNGramSize
   *          1 = unigram, 2 = unigram and bigram, 3 = unigram, bigrama and
   *          trigram
   * @param minLLRValue
   *          minValue of log likelihood ratio to used to prune ngrams
   * @param chunkSizeInMegabytes
   *          the size in MB of the feature => id chunk to be kept in memory at
   *          each node during Map/Reduce stage. Its recommended you calculated
   *          this based on the number of cores and the free memory available to
   *          you per node. Say, you have 2 cores and around 1GB extra memory to
   *          spare we recommend you use a split size of around 400-500MB so
   *          that two simultaneous reducers can create partial vectors without
   *          thrashing the system due to increased swapping
   * @throws IOException
   */
  public static void createTermFrequencyVectors(String input,
                                                String output,
                                                int minSupport,
                                                int maxNGramSize,
                                                float minLLRValue,
                                                int numReducers,
                                                int chunkSizeInMegabytes) throws IOException {
    if (chunkSizeInMegabytes < MIN_CHUNKSIZE) {
      chunkSizeInMegabytes = MIN_CHUNKSIZE;
    } else if (chunkSizeInMegabytes > MAX_CHUNKSIZE) { // 10GB
      chunkSizeInMegabytes = MAX_CHUNKSIZE;
    }
    if (minSupport < 0) minSupport = DEFAULT_MIN_SUPPORT;
    
    Path inputPath = new Path(input);
    Path dictionaryJobPath = new Path(output + DICTIONARY_JOB_FOLDER);
    
    List<Path> dictionaryChunks;
    if (maxNGramSize == 1) {
      startWordCounting(inputPath, dictionaryJobPath, minSupport);
      dictionaryChunks = createDictionaryChunks(minSupport, dictionaryJobPath,
        output, chunkSizeInMegabytes, new LongWritable());
    } else {
      CollocDriver.generateAllGrams(inputPath.toString(), dictionaryJobPath
          .toString(), maxNGramSize, minSupport, minLLRValue, numReducers);
      dictionaryChunks = createDictionaryChunks(minSupport, new Path(
          output + DICTIONARY_JOB_FOLDER + "/ngrams"), output,
        chunkSizeInMegabytes, new DoubleWritable());
    }
    
    int partialVectorIndex = 0;
    List<Path> partialVectorPaths = new ArrayList<Path>();
    for (Path dictionaryChunk : dictionaryChunks) {
      Path partialVectorOutputPath = getPath(output + VECTOR_OUTPUT_FOLDER,
        partialVectorIndex++);
      partialVectorPaths.add(partialVectorOutputPath);
      makePartialVectors(input, maxNGramSize, dictionaryChunk,
        partialVectorOutputPath);
    }
    
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(partialVectorPaths.get(0).toUri(), conf);
    
    String outputDir = output + DOCUMENT_VECTOR_OUTPUT_FOLDER;
    if (dictionaryChunks.size() > 1) {
      PartialVectorMerger
          .mergePartialVectors(partialVectorPaths, outputDir, -1);
      HadoopUtil.deletePaths(partialVectorPaths, fs);
    } else {
      Path singlePartialVectorOutputPath = partialVectorPaths.get(0);
      HadoopUtil.deletePath(outputDir, fs);
      HadoopUtil.rename(singlePartialVectorOutputPath, new Path(outputDir), fs);
    }
  }
  
  /**
   * Read the feature frequency List which is built at the end of the Word Count
   * Job and assign ids to them. This will use constant memory and will run at
   * the speed of your disk read
   * 
   * @param minSupport
   * @param wordCountPath
   * @param dictionaryPathBase
   * @throws IOException
   */
  private static List<Path> createDictionaryChunks(int minSupport,
                                                   Path wordCountPath,
                                                   String dictionaryPathBase,
                                                   int chunkSizeInMegabytes,
                                                   Writable value) throws IOException {
    List<Path> chunkPaths = new ArrayList<Path>();
    
    Writable key = new Text();
    Configuration conf = new Configuration();
    
    FileSystem fs = FileSystem.get(wordCountPath.toUri(), conf);
    FileStatus[] outputFiles = fs.globStatus(new Path(wordCountPath.toString()
                                                      + OUTPUT_FILES_PATTERN));
    
    long chunkSizeLimit = chunkSizeInMegabytes * 1024 * 1024;
    int chunkIndex = 0;
    Path chunkPath = getPath(dictionaryPathBase + DICTIONARY_FILE, chunkIndex);
    chunkPaths.add(chunkPath);
    
    SequenceFile.Writer dictWriter = new SequenceFile.Writer(fs, conf,
        chunkPath, Text.class, IntWritable.class);
    
    long currentChunkSize = 0;
    
    int i = 0;
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // key is feature value is count
      while (reader.next(key, value)) {
        if (currentChunkSize > chunkSizeLimit) {
          dictWriter.close();
          chunkIndex++;
          
          chunkPath = getPath(dictionaryPathBase + DICTIONARY_FILE, chunkIndex);
          chunkPaths.add(chunkPath);
          
          dictWriter = new SequenceFile.Writer(fs, conf, chunkPath, Text.class,
              IntWritable.class);
          currentChunkSize = 0;
        }
        
        int fieldSize = DICTIONARY_BYTE_OVERHEAD
                        + (key.toString().length() * 2) + (Integer.SIZE / 8);
        currentChunkSize += fieldSize;
        dictWriter.append(key, new IntWritable(i++));
      }
    }
    
    dictWriter.close();
    
    return chunkPaths;
  }
  
  private static Path getPath(String basePath, int index) {
    return new Path(basePath + index);
  }
  
  /**
   * Create a partial vector using a chunk of features from the input documents.
   * The input documents has to be in the {@link SequenceFile} format
   * 
   * @param input
   *          input directory of the documents in {@link SequenceFile} format
   * @param maxNGramSize
   *          maximum size of ngrams to generate
   * @param dictionaryFilePath
   *          location of the chunk of features and the id's
   * @param output
   *          output directory were the partial vectors have to be created
   * @throws IOException
   */
  private static void makePartialVectors(String input,
                                         int maxNGramSize,
                                         Path dictionaryFilePath,
                                         Path output) throws IOException {
    
    Configurable client = new JobClient();
    JobConf conf = new JobConf(DictionaryVectorizer.class);
    conf.set("io.serializations",
      "org.apache.hadoop.io.serializer.JavaSerialization,"
          + "org.apache.hadoop.io.serializer.WritableSerialization");
    // this conf parameter needs to be set enable serialisation of conf values
    
    conf.setJobName("DictionaryVectorizer::MakePartialVectors: input-folder: "
                    + input + ", dictionary-file: "
                    + dictionaryFilePath.toString());
    conf.setInt(MAX_NGRAMS, maxNGramSize);
    
    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(StringTuple.class);
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(VectorWritable.class);
    DistributedCache
        .setCacheFiles(new URI[] {dictionaryFilePath.toUri()}, conf);
    FileInputFormat.setInputPaths(conf, new Path(input));
    
    FileOutputFormat.setOutputPath(conf, output);
    
    conf.setMapperClass(IdentityMapper.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setReducerClass(TFPartialVectorReducer.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    FileSystem dfs = FileSystem.get(output.toUri(), conf);
    if (dfs.exists(output)) {
      dfs.delete(output, true);
    }
    
    client.setConf(conf);
    JobClient.runJob(conf);
  }
  
  /**
   * Count the frequencies of words in parallel using Map/Reduce. The input
   * documents have to be in {@link SequenceFile} format
   */
  private static void startWordCounting(Path input, Path output, int minSupport) throws IOException {
    
    Configurable client = new JobClient();
    JobConf conf = new JobConf(DictionaryVectorizer.class);
    conf.set("io.serializations",
      "org.apache.hadoop.io.serializer.JavaSerialization,"
          + "org.apache.hadoop.io.serializer.WritableSerialization");
    // this conf parameter needs to be set enable serialisation of conf values
    
    conf.setJobName("DictionaryVectorizer::WordCount: input-folder: "
                    + input.toString());
    conf.setInt(MIN_SUPPORT, minSupport);
    
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(LongWritable.class);
    
    FileInputFormat.setInputPaths(conf, input);
    Path outPath = output;
    FileOutputFormat.setOutputPath(conf, outPath);
    
    conf.setMapperClass(TermCountMapper.class);
    
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setCombinerClass(TermCountReducer.class);
    conf.setReducerClass(TermCountReducer.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    
    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }
    
    client.setConf(conf);
    JobClient.runJob(conf);
  }
}
