/*
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

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
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
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.pruner.PrunedPartialVectorMergeReducer;
import org.apache.mahout.vectorizer.pruner.WordsPrunerReducer;

import java.io.IOException;
import java.net.URI;
import java.util.List;

public final class HighDFWordsPruner {

  public static final String STD_CALC_DIR = "stdcalc";
  public static final String MAX_DF = "max.df";
  public static final String MIN_DF = "min.df";

  private HighDFWordsPruner() {
  }

  public static void pruneVectors(Path tfDir, Path prunedTFDir, Path prunedPartialTFDir, long maxDF,
                                  long minDF, Configuration baseConf,
                                  Pair<Long[], List<Path>> docFrequenciesFeatures,
                                  float normPower,
                                  boolean logNormalize,
                                  int numReducers) throws IOException, InterruptedException, ClassNotFoundException {

    int partialVectorIndex = 0;
    List<Path> partialVectorPaths = Lists.newArrayList();
    for (Path path : docFrequenciesFeatures.getSecond()) {
      Path partialVectorOutputPath = new Path(prunedPartialTFDir, "partial-" + partialVectorIndex++);
      partialVectorPaths.add(partialVectorOutputPath);
      pruneVectorsPartial(tfDir, partialVectorOutputPath, path, maxDF, minDF, baseConf);
    }

    mergePartialVectors(partialVectorPaths, prunedTFDir, baseConf, normPower, logNormalize, numReducers);
    HadoopUtil.delete(new Configuration(baseConf), prunedPartialTFDir);
  }

  private static void pruneVectorsPartial(Path input, Path output, Path dictionaryFilePath, long maxDF,
                                          long minDF, Configuration baseConf) throws IOException, InterruptedException,
          ClassNotFoundException {

    Configuration conf = new Configuration(baseConf);
    // this conf parameter needs to be set enable serialisation of conf
    // values
    conf.set("io.serializations",
            "org.apache.hadoop.io.serializer.JavaSerialization,"
                    + "org.apache.hadoop.io.serializer.WritableSerialization");
    conf.setLong(MAX_DF, maxDF);
    conf.setLong(MIN_DF, minDF);
    DistributedCache.setCacheFiles(
            new URI[]{dictionaryFilePath.toUri()}, conf);

    Job job = HadoopUtil.prepareJob(input, output, SequenceFileInputFormat.class,
            Mapper.class, null, null, WordsPrunerReducer.class,
            Text.class, VectorWritable.class, SequenceFileOutputFormat.class,
            conf);
    job.setJobName(": Prune Vectors: input-folder: " + input
            + ", dictionary-file: " + dictionaryFilePath.toString());

    HadoopUtil.delete(conf, output);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }

  public static void mergePartialVectors(Iterable<Path> partialVectorPaths,
                                         Path output,
                                         Configuration baseConf,
                                         float normPower,
                                         boolean logNormalize,
                                         int numReducers)
    throws IOException, InterruptedException, ClassNotFoundException {

    Configuration conf = new Configuration(baseConf);
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
            + "org.apache.hadoop.io.serializer.WritableSerialization");
    conf.setFloat(PartialVectorMerger.NORMALIZATION_POWER, normPower);
    conf.setBoolean(PartialVectorMerger.LOG_NORMALIZE, logNormalize);

    Job job = new Job(conf);
    job.setJobName("PrunerPartialVectorMerger::MergePartialVectors");
    job.setJarByClass(PartialVectorMerger.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(VectorWritable.class);

    FileInputFormat.setInputPaths(job, getCommaSeparatedPaths(partialVectorPaths));

    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(Mapper.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setReducerClass(PrunedPartialVectorMergeReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setNumReduceTasks(numReducers);

    HadoopUtil.delete(conf, output);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }

  private static String getCommaSeparatedPaths(Iterable<Path> paths) {
    StringBuilder commaSeparatedPaths = new StringBuilder(100);
    String sep = "";
    for (Path path : paths) {
      commaSeparatedPaths.append(sep).append(path.toString());
      sep = ",";
    }
    return commaSeparatedPaths.toString();
  }
}
