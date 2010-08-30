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

package org.apache.mahout.utils.vectors.tfidf;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
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
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.common.PartialVectorMerger;
import org.apache.mahout.utils.vectors.text.term.TermDocumentCountMapper;
import org.apache.mahout.utils.vectors.text.term.TermDocumentCountReducer;

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

  public static final String MAX_DF_PERCENTAGE = "max.df.percentage";

  //public static final String TFIDF_OUTPUT_FOLDER = "tfidf";

  private static final String DOCUMENT_VECTOR_OUTPUT_FOLDER = "tfidf-vectors";

  private static final String FREQUENCY_FILE = "frequency.file-";

  private static final int MAX_CHUNKSIZE = 10000;

  private static final int MIN_CHUNKSIZE = 100;

  private static final String OUTPUT_FILES_PATTERN = "part-*";

  private static final int SEQUENCEFILE_BYTE_OVERHEAD = 45;

  private static final String VECTOR_OUTPUT_FOLDER = "partial-vectors-";

  private static final String WORDCOUNT_OUTPUT_FOLDER = "df-count";

  /**
   * Cannot be initialized. Use the static functions
   */
  private TFIDFConverter() {

  }

  /**
   * Create Term Frequency-Inverse Document Frequency (Tf-Idf) Vectors from the input set of vectors in
   * {@link SequenceFile} format. This job uses a fixed limit on the maximum memory used by the feature chunk
   * per node thereby splitting the process across multiple map/reduces.
   * 
   * @param input
   *          input directory of the vectors in {@link SequenceFile} format
   * @param output
   *          output directory where {@link org.apache.mahout.math.RandomAccessSparseVector}'s of the document
   *          are generated
   * @param chunkSizeInMegabytes
   *          the size in MB of the feature => id chunk to be kept in memory at each node during Map/Reduce
   *          stage. Its recommended you calculated this based on the number of cores and the free memory
   *          available to you per node. Say, you have 2 cores and around 1GB extra memory to spare we
   *          recommend you use a split size of around 400-500MB so that two simultaneous reducers can create
   *          partial vectors without thrashing the system due to increased swapping
   * @param minDf
   *          The minimum document frequency. Default 1
   * @param maxDFPercent
   *          The max percentage of vectors for the DF. Can be used to remove really high frequency features.
   *          Expressed as an integer between 0 and 100. Default 99
   * @param numReducers 
   *          The number of reducers to spawn. This also affects the possible parallelism since each reducer
   *          will typically produce a single output file containing tf-idf vectors for a subset of the
   *          documents in the corpus.
   */
  public static void processTfIdf(Path input,
                                  Path output,
                                  int chunkSizeInMegabytes,
                                  int minDf,
                                  int maxDFPercent,
                                  float normPower,
                                  boolean sequentialAccessOutput,
                                  int numReducers) throws IOException, InterruptedException, ClassNotFoundException {
    if (chunkSizeInMegabytes < MIN_CHUNKSIZE) {
      chunkSizeInMegabytes = MIN_CHUNKSIZE;
    } else if (chunkSizeInMegabytes > MAX_CHUNKSIZE) { // 10GB
      chunkSizeInMegabytes = MAX_CHUNKSIZE;
    }

    if (normPower != PartialVectorMerger.NO_NORMALIZING && normPower < 0) {
      throw new IllegalArgumentException("normPower must either be -1 or >= 0");
    }

    if (minDf < 1) {
      minDf = 1;
    }
    if (maxDFPercent < 0 || maxDFPercent > 100) {
      maxDFPercent = 99;
    }

    Path wordCountPath = new Path(output, WORDCOUNT_OUTPUT_FOLDER);

    startDFCounting(input, wordCountPath);
    Pair<Long[], List<Path>> datasetFeatures = createDictionaryChunks(wordCountPath, output, chunkSizeInMegabytes);

    int partialVectorIndex = 0;
    List<Path> partialVectorPaths = new ArrayList<Path>();
    List<Path> dictionaryChunks = datasetFeatures.getSecond();
    for (Path dictionaryChunk : dictionaryChunks) {
      Path partialVectorOutputPath = new Path(output, VECTOR_OUTPUT_FOLDER + partialVectorIndex++);
      partialVectorPaths.add(partialVectorOutputPath);
      makePartialVectors(input,
                         datasetFeatures.getFirst()[0],
                         datasetFeatures.getFirst()[1],
                         minDf,
                         maxDFPercent,
                         dictionaryChunk,
                         partialVectorOutputPath,
                         sequentialAccessOutput);
    }

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(partialVectorPaths.get(0).toUri(), conf);

    Path outputDir = new Path(output, DOCUMENT_VECTOR_OUTPUT_FOLDER);
    if (dictionaryChunks.size() > 1) {
      PartialVectorMerger.mergePartialVectors(partialVectorPaths,
                                              outputDir,
                                              normPower,
                                              datasetFeatures.getFirst()[0].intValue(),
                                              sequentialAccessOutput,
                                              numReducers);
      HadoopUtil.deletePaths(partialVectorPaths, fs);
    } else {
      Path singlePartialVectorOutputPath = partialVectorPaths.get(0);
      fs.delete(outputDir, true);
      fs.rename(singlePartialVectorOutputPath, outputDir);
    }
  }

  /**
   * Read the document frequency List which is built at the end of the DF Count Job. This will use constant
   * memory and will run at the speed of your disk read
   */
  private static Pair<Long[], List<Path>> createDictionaryChunks(Path featureCountPath,
                                                                 Path dictionaryPathBase,
                                                                 int chunkSizeInMegabytes) throws IOException {
    List<Path> chunkPaths = new ArrayList<Path>();

    IntWritable key = new IntWritable();
    LongWritable value = new LongWritable();
    Configuration conf = new Configuration();

    FileSystem fs = FileSystem.get(featureCountPath.toUri(), conf);
    FileStatus[] outputFiles = fs.globStatus(new Path(featureCountPath, OUTPUT_FILES_PATTERN));

    long chunkSizeLimit = chunkSizeInMegabytes * 1024L * 1024L;
    int chunkIndex = 0;
    Path chunkPath = new Path(dictionaryPathBase, FREQUENCY_FILE + chunkIndex);
    chunkPaths.add(chunkPath);
    SequenceFile.Writer freqWriter =
      new SequenceFile.Writer(fs, conf, chunkPath, IntWritable.class, LongWritable.class);

    long currentChunkSize = 0;
    long featureCount = 0;
    long vectorCount = Long.MAX_VALUE;
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // key is feature value is count
      while (reader.next(key, value)) {
        if (currentChunkSize > chunkSizeLimit) {
          freqWriter.close();
          chunkIndex++;

          chunkPath = new Path(dictionaryPathBase, FREQUENCY_FILE + chunkIndex);
          chunkPaths.add(chunkPath);

          freqWriter = new SequenceFile.Writer(fs, conf, chunkPath, IntWritable.class, LongWritable.class);
          currentChunkSize = 0;
        }

        int fieldSize = SEQUENCEFILE_BYTE_OVERHEAD + Integer.SIZE / 8 + Long.SIZE / 8;
        currentChunkSize += fieldSize;
        if (key.get() >= 0) {
          freqWriter.append(key, value);
        } else if (key.get() == -1) {
          vectorCount = value.get();
        }
        featureCount = Math.max(key.get(), featureCount);

      }
    }
    featureCount++;
    freqWriter.close();
    Long[] counts = {featureCount, vectorCount};
    return new Pair<Long[], List<Path>>(counts, chunkPaths);
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
   * @param maxDFPercent
   *          The max percentage of vectors for the DF. Can be used to remove really high frequency features.
   *          Expressed as an integer between 0 and 100. Default 99
   * @param dictionaryFilePath
   *          location of the chunk of features and the id's
   * @param output
   *          output directory were the partial vectors have to be created
   */
  private static void makePartialVectors(Path input,
                                         Long featureCount,
                                         Long vectorCount,
                                         int minDf,
                                         int maxDFPercent,
                                         Path dictionaryFilePath,
                                         Path output,
                                         boolean sequentialAccess)
    throws IOException, InterruptedException, ClassNotFoundException {

    Configuration conf = new Configuration();
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    conf.setLong(FEATURE_COUNT, featureCount);
    conf.setLong(VECTOR_COUNT, vectorCount);
    conf.setInt(MIN_DF, minDf);
    conf.setInt(MAX_DF_PERCENTAGE, maxDFPercent);
    conf.setBoolean(PartialVectorMerger.SEQUENTIAL_ACCESS, sequentialAccess);
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

    HadoopUtil.overwriteOutput(output);

    job.waitForCompletion(true);
  }

  /**
   * Count the document frequencies of features in parallel using Map/Reduce. The input documents have to be
   * in {@link SequenceFile} format
   */
  private static void startDFCounting(Path input, Path output)
    throws IOException, InterruptedException, ClassNotFoundException {

    Configuration conf = new Configuration();
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    
    Job job = new Job(conf);
    job.setJobName("VectorTfIdf Document Frequency Count running over input: " + input.toString());
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

    HadoopUtil.overwriteOutput(output);

    job.waitForCompletion(true);
  }
}
