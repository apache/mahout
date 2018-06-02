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

package org.apache.mahout.clustering.spectral;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

public final class AffinityMatrixInputJob {

  private AffinityMatrixInputJob() {
  }

  /**
   * Initializes and executes the job of reading the documents containing
   * the data of the affinity matrix in (x_i, x_j, value) format.
   */
  public static void runJob(Path input, Path output, int rows, int cols)
    throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    HadoopUtil.delete(conf, output);

    conf.setInt(Keys.AFFINITY_DIMENSIONS, rows);
    Job job = new Job(conf, "AffinityMatrixInputJob: " + input + " -> M/R -> " + output);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(DistributedRowMatrix.MatrixEntryWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(AffinityMatrixInputMapper.class);   
    job.setReducerClass(AffinityMatrixInputReducer.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setJarByClass(AffinityMatrixInputJob.class);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }

  /**
   * A transparent wrapper for the above method which handles the tedious tasks
   * of setting and retrieving system Paths. Hands back a fully-populated
   * and initialized DistributedRowMatrix.
   */
  public static DistributedRowMatrix runJob(Path input, Path output, int dimensions)
    throws IOException, InterruptedException, ClassNotFoundException {
    Path seqFiles = new Path(output, "seqfiles-" + (System.nanoTime() & 0xFF));
    runJob(input, seqFiles, dimensions, dimensions);
    DistributedRowMatrix a = new DistributedRowMatrix(seqFiles,
        new Path(seqFiles, "seqtmp-" + (System.nanoTime() & 0xFF)), 
        dimensions, dimensions);
    a.setConf(new Configuration());
    return a;
  }
}
