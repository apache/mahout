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

package org.apache.mahout.vectorizer.common;

import java.io.IOException;

import com.google.common.base.Preconditions;
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
  
  public static final String NAMED_VECTOR = "vector.named";

  public static final String LOG_NORMALIZE = "vector.lognormalize";

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
   * @param baseConf
   *          job configuration
   * @param normPower
   *          The normalization value. Must be greater than or equal to 0 or equal to {@link #NO_NORMALIZING}
   * @param dimension cardinality of the vectors
   * @param sequentialAccess
   *          output vectors should be optimized for sequential access
   * @param namedVector
   *          output vectors should be named, retaining key (doc id) as a label
   * @param numReducers 
   *          The number of reducers to spawn
   */
  public static void mergePartialVectors(Iterable<Path> partialVectorPaths,
                                         Path output,
                                         Configuration baseConf,
                                         float normPower,
                                         boolean logNormalize,
                                         int dimension,
                                         boolean sequentialAccess,
                                         boolean namedVector,
                                         int numReducers)
    throws IOException, InterruptedException, ClassNotFoundException {
    Preconditions.checkArgument(normPower == NO_NORMALIZING || normPower >= 0,
        "If specified normPower must be nonnegative", normPower);
    Preconditions.checkArgument(normPower == NO_NORMALIZING
                                || (normPower > 1 && !Double.isInfinite(normPower))
                                || !logNormalize,
        "normPower must be > 1 and not infinite if log normalization is chosen", normPower);

    Configuration conf = new Configuration(baseConf);
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    conf.setBoolean(SEQUENTIAL_ACCESS, sequentialAccess);
    conf.setBoolean(NAMED_VECTOR, namedVector);
    conf.setInt(DIMENSION, dimension);
    conf.setFloat(NORMALIZATION_POWER, normPower);
    conf.setBoolean(LOG_NORMALIZE, logNormalize);

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
