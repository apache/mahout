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

package org.apache.mahout.utils.vectors.common;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.VectorWritable;

/**
 * This class groups a set of input vectors. The Sequence file input should have a
 * {@link org.apache.hadoop.io.WritableComparable}
 * key containing document id and a {@link VectorWritable} value containing the term frequency vector. This
 * class also does normalization of the vector.
 * 
 */
public final class PartialVectorMerger {

  public static final float NO_NORMALIZING = -1.0f;

  public static final String NORMALIZATION_POWER = "normalization.power";

  public static final String DIMENSION = "vector.dimension";

  public static final String SEQUENTIAL_ACCESS = "vector.sequentialAccess";

  /**
   * Cannot be initialized. Use the static functions
   */
  private PartialVectorMerger() {

  }

  /**
   * Merge all the partial {@link org.apache.mahout.math.RandomAccessSparseVector}s into the complete Document
   * {@link org.apache.mahout.math.RandomAccessSparseVector}
   * 
   * @param partialVectorPaths
   *          input directory of the vectors in {@link org.apache.hadoop.io.SequenceFile} format
   * @param output
   *          output directory were the partial vectors have to be created
   * @param normPower
   *          The normalization value. Must be greater than or equal to 0 or equal to {@link #NO_NORMALIZING}
   * @param numReducers 
   *          The number of reducers to spawn
   * @throws IOException
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public static void mergePartialVectors(Iterable<Path> partialVectorPaths,
                                         Path output,
                                         float normPower,
                                         int dimension,
                                         boolean sequentialAccess,
                                         int numReducers) throws IOException, InterruptedException, ClassNotFoundException {
    if (normPower != NO_NORMALIZING && normPower < 0) {
      throw new IllegalArgumentException("normPower must either be -1 or >= 0");
    }

    Configuration conf = new Configuration();
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    conf.setBoolean(SEQUENTIAL_ACCESS, sequentialAccess);
    conf.setInt(DIMENSION, dimension);
    conf.setFloat(NORMALIZATION_POWER, normPower);

    Job job = new Job(conf);
    job.setJobName("PartialVectorMerger::MergePartialVectors");
    job.setJarByClass(PartialVectorMerger.class);
    
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(VectorWritable.class);

    FileInputFormat.setInputPaths(job, getCommaSeparatedPaths(partialVectorPaths));

    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(Mapper.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setReducerClass(PartialVectorMergeReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setNumReduceTasks(numReducers);

    HadoopUtil.overwriteOutput(output);

    job.waitForCompletion(true);
  }

  private static String getCommaSeparatedPaths(Iterable<Path> paths) {
    StringBuilder commaSeparatedPaths = new StringBuilder();
    String sep = "";
    for (Path path : paths) {
      commaSeparatedPaths.append(sep).append(path.toString());
      sep = ",";
    }
    return commaSeparatedPaths.toString();
  }
}
