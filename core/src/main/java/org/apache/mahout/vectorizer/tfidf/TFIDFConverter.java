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

package org.apache.mahout.vectorizer.tfidf;

import java.io.IOException;
import java.net.URI;
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
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.term.TermDocumentCountMapper;
import org.apache.mahout.vectorizer.term.TermDocumentCountReducer;

/**
 * This class converts a set of input vectors with term frequencies to TfIdf vectors. The Sequence file input
 * should have a {@link org.apache.hadoop.io.WritableComparable} key containing and a
 * {@link VectorWritable} value containing the
 * term frequency vector. This is conversion class uses multiple map/reduces to convert the vectors to TfIdf
 * format
 * 
 */
public final class TFIDFConverter {

  public static final String VECTOR_COUNT = "vector.count";
  public static final String FEATURE_COUNT = "feature.count";
  public static final String MIN_DF = "min.df";
  public static final String MAX_DF = "max.df";
  //public static final String TFIDF_OUTPUT_FOLDER = "tfidf";

  private static final String DOCUMENT_VECTOR_OUTPUT_FOLDER = "tfidf-vectors";
  private static final String FREQUENCY_FILE = "frequency.file-";
  private static final int MAX_CHUNKSIZE = 10000;
  private static final int MIN_CHUNKSIZE = 100;
  private static final String OUTPUT_FILES_PATTERN = "part-*";
  private static final int SEQUENCEFILE_BYTE_OVERHEAD = 45;
  private static final String VECTOR_OUTPUT_FOLDER = "partial-vectors-";
  public static final String WORDCOUNT_OUTPUT_FOLDER = "df-count";

  /**
   * Cannot be initialized. Use the static functions
   */
  private TFIDFConverter() {
  }

  /**
   * Create Term Frequency-Inverse Document Frequency (Tf-Idf) Vectors from the input set of vectors in
   * {@link SequenceFile} format. This job uses a fixed limit on the maximum memory used by the feature chunk
   * per node thereby splitting the process across multiple map/reduces.
   * Before using this method calculateDF should be called
   * 
   * @param input
   *          input directory of the vectors in {@link SequenceFile} format
   * @param output
   *          output directory where {@link org.apache.mahout.math.RandomAccessSparseVector}'s of the document
   *          are generated
   * @param datasetFeatures
   *          Document frequencies information calculated by calculateDF
   * @param minDf
   *          The minimum document frequency. Default 1
   * @param maxDF
   *          The max percentage of vectors for the DF. Can be used to remove really high frequency features.
   *          Expressed as an integer between 0 and 100. Default 99
   * @param numReducers 
   *          The number of reducers to spawn. This also affects the possible parallelism since each reducer
   *          will typically produce a single output file containing tf-idf vectors for a subset of the
   *          documents in the corpus.
   */
  public static void processTfIdf(Path input,
                                  Path output,
                                  Configuration baseConf,
                                  Pair<Long[], List<Path>> datasetFeatures,
                                  int minDf,
                                  long maxDF,
                                  float normPower,
                                  boolean logNormalize,
                                  boolean sequentialAccessOutput,
                                  boolean namedVector,
                                  int numReducers) throws IOException, InterruptedException, ClassNotFoundException {
    Preconditions.checkArgument(normPower == PartialVectorMerger.NO_NORMALIZING || normPower >= 0,
        "If specified normPower must be nonnegative", normPower);
    Preconditions.checkArgument(normPower == PartialVectorMerger.NO_NORMALIZING
                                || (normPower > 1 && !Double.isInfinite(normPower))
                                || !logNormalize,
        "normPower must be > 1 and not infinite if log normalization is chosen", normPower);

    int partialVectorIndex = 0;
    List<Path> partialVectorPaths = Lists.newArrayList();
    List<Path> dictionaryChunks = datasetFeatures.getSecond();
    for (Path dictionaryChunk : dictionaryChunks) {
      Path partialVectorOutputPath = new Path(output, VECTOR_OUTPUT_FOLDER + partialVectorIndex++);
      partialVectorPaths.add(partialVectorOutputPath);
      makePartialVectors(input,
                         baseConf,
                         datasetFeatures.getFirst()[0],
                         datasetFeatures.getFirst()[1],
                         minDf,
                         maxDF,
                         dictionaryChunk,
                         partialVectorOutputPath,
                         sequentialAccessOutput,
                         namedVector);
    }

    Configuration conf = new Configuration(baseConf);

    Path outputDir = new Path(output, DOCUMENT_VECTOR_OUTPUT_FOLDER);
    
    PartialVectorMerger.mergePartialVectors(partialVectorPaths,
                                            outputDir,
                                            baseConf,
                                            normPower,
                                            logNormalize,
                                            datasetFeatures.getFirst()[0].intValue(),
                                            sequentialAccessOutput,
                                            namedVector,
                                            numReducers);
    HadoopUtil.delete(conf, partialVectorPaths);

  }
  
  /**
   * Calculates the document frequencies of all terms from the input set of vectors in
   * {@link SequenceFile} format. This job uses a fixed limit on the maximum memory used by the feature chunk
   * per node thereby splitting the process across multiple map/reduces.
   * 
   * @param input
   *          input directory of the vectors in {@link SequenceFile} format
   * @param output
   *          output directory where document frequencies will be stored
   * @param chunkSizeInMegabytes
   *          the size in MB of the feature => id chunk to be kept in memory at each node during Map/Reduce
   *          stage. Its recommended you calculated this based on the number of cores and the free memory
   *          available to you per node. Say, you have 2 cores and around 1GB extra memory to spare we
   *          recommend you use a split size of around 400-500MB so that two simultaneous reducers can create
   *          partial vectors without thrashing the system due to increased swapping
   */
  public static Pair<Long[],List<Path>> calculateDF(Path input,
                                                    Path output,
                                                    Configuration baseConf,
                                                    int chunkSizeInMegabytes)
    throws IOException, InterruptedException, ClassNotFoundException {

    if (chunkSizeInMegabytes < MIN_CHUNKSIZE) {
      chunkSizeInMegabytes = MIN_CHUNKSIZE;
    } else if (chunkSizeInMegabytes > MAX_CHUNKSIZE) { // 10GB
      chunkSizeInMegabytes = MAX_CHUNKSIZE;
    }

    Path wordCountPath = new Path(output, WORDCOUNT_OUTPUT_FOLDER);

    startDFCounting(input, wordCountPath, baseConf);

    return createDictionaryChunks(wordCountPath, output, baseConf, chunkSizeInMegabytes);
  }

  /**
   * Read the document frequency List which is built at the end of the DF Count Job. This will use constant
   * memory and will run at the speed of your disk read
   */
  private static Pair<Long[], List<Path>> createDictionaryChunks(Path featureCountPath,
                                                                 Path dictionaryPathBase,
                                                                 Configuration baseConf,
                                                                 int chunkSizeInMegabytes) throws IOException {
    List<Path> chunkPaths = Lists.newArrayList();
    Configuration conf = new Configuration(baseConf);

    FileSystem fs = FileSystem.get(featureCountPath.toUri(), conf);

    long chunkSizeLimit = chunkSizeInMegabytes * 1024L * 1024L;
    int chunkIndex = 0;
    Path chunkPath = new Path(dictionaryPathBase, FREQUENCY_FILE + chunkIndex);
    chunkPaths.add(chunkPath);
    SequenceFile.Writer freqWriter =
      new SequenceFile.Writer(fs, conf, chunkPath, IntWritable.class, LongWritable.class);

    try {
      long currentChunkSize = 0;
      long featureCount = 0;
      long vectorCount = Long.MAX_VALUE;
      Path filesPattern = new Path(featureCountPath, OUTPUT_FILES_PATTERN);
      for (Pair<IntWritable,LongWritable> record
           : new SequenceFileDirIterable<IntWritable,LongWritable>(filesPattern,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {

        if (currentChunkSize > chunkSizeLimit) {
          Closeables.close(freqWriter, false);
          chunkIndex++;

          chunkPath = new Path(dictionaryPathBase, FREQUENCY_FILE + chunkIndex);
          chunkPaths.add(chunkPath);

          freqWriter = new SequenceFile.Writer(fs, conf, chunkPath, IntWritable.class, LongWritable.class);
          currentChunkSize = 0;
        }

        int fieldSize = SEQUENCEFILE_BYTE_OVERHEAD + Integer.SIZE / 8 + Long.SIZE / 8;
        currentChunkSize += fieldSize;
        IntWritable key = record.getFirst();
        LongWritable value = record.getSecond();
        if (key.get() >= 0) {
          freqWriter.append(key, value);
        } else if (key.get() == -1) {
          vectorCount = value.get();
        }
        featureCount = Math.max(key.get(), featureCount);

      }
      featureCount++;
      Long[] counts = {featureCount, vectorCount};
      return new Pair<Long[], List<Path>>(counts, chunkPaths);
    } finally {
      Closeables.close(freqWriter, false);
    }
  }

  /**
   * Create a partial tfidf vector using a chunk of features from the input vectors. The input vectors has to
   * be in the {@link SequenceFile} format
   * 
   * @param input
   *          input directory of the vectors in {@link SequenceFile} format
   * @param featureCount
   *          Number of unique features in the dataset
   * @param vectorCount
   *          Number of vectors in the dataset
   * @param minDf
   *          The minimum document frequency. Default 1
   * @param maxDF
   *          The max percentage of vectors for the DF. Can be used to remove really high frequency features.
   *          Expressed as an integer between 0 and 100. Default 99
   * @param dictionaryFilePath
   *          location of the chunk of features and the id's
   * @param output
   *          output directory were the partial vectors have to be created
   * @param sequentialAccess
   *          output vectors should be optimized for sequential access
   * @param namedVector
   *          output vectors should be named, retaining key (doc id) as a label
   */
  private static void makePartialVectors(Path input,
                                         Configuration baseConf,
                                         Long featureCount,
                                         Long vectorCount,
                                         int minDf,
                                         long maxDF,
                                         Path dictionaryFilePath,
                                         Path output,
                                         boolean sequentialAccess,
                                         boolean namedVector)
    throws IOException, InterruptedException, ClassNotFoundException {

    Configuration conf = new Configuration(baseConf);
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    conf.setLong(FEATURE_COUNT, featureCount);
    conf.setLong(VECTOR_COUNT, vectorCount);
    conf.setInt(MIN_DF, minDf);
    conf.setLong(MAX_DF, maxDF);
    conf.setBoolean(PartialVectorMerger.SEQUENTIAL_ACCESS, sequentialAccess);
    conf.setBoolean(PartialVectorMerger.NAMED_VECTOR, namedVector);
    DistributedCache.setCacheFiles(new URI[] {dictionaryFilePath.toUri()}, conf);

    Job job = new Job(conf);
    job.setJobName(": MakePartialVectors: input-folder: " + input + ", dictionary-file: "
        + dictionaryFilePath.toString());
    job.setJarByClass(TFIDFConverter.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(VectorWritable.class);
    FileInputFormat.setInputPaths(job, input);

    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(Mapper.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setReducerClass(TFIDFPartialVectorReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    HadoopUtil.delete(conf, output);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }

  /**
   * Count the document frequencies of features in parallel using Map/Reduce. The input documents have to be
   * in {@link SequenceFile} format
   */
  private static void startDFCounting(Path input, Path output, Configuration baseConf)
    throws IOException, InterruptedException, ClassNotFoundException {

    Configuration conf = new Configuration(baseConf);
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    
    Job job = new Job(conf);
    job.setJobName("VectorTfIdf Document Frequency Count running over input: " + input);
    job.setJarByClass(TFIDFConverter.class);
    
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(LongWritable.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(TermDocumentCountMapper.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setCombinerClass(TermDocumentCountReducer.class);
    job.setReducerClass(TermDocumentCountReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    HadoopUtil.delete(conf, output);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }
}
