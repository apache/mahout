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
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.VectorWritable;

/**
 * <p>Given a DistributedRowMatrix, this job normalizes each row to unit
 * vector length. If the input is a matrix U, and the output is a matrix
 * W, the job follows:</p>
 *
 * <p>{@code v_ij = u_ij / sqrt(sum_j(u_ij * u_ij))}</p>
 */
public final class UnitVectorizerJob {

  private UnitVectorizerJob() {
  }

  public static void runJob(Path input, Path output)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    Configuration conf = new Configuration();
    Job job = new Job(conf, "UnitVectorizerJob");
    
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(UnitVectorizerMapper.class);
    job.setNumReduceTasks(0);
    
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setJarByClass(UnitVectorizerJob.class);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }
  
  public static class UnitVectorizerMapper
    extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    
    @Override
    protected void map(IntWritable row, VectorWritable vector, Context context) 
      throws IOException, InterruptedException {
      context.write(row, new VectorWritable(vector.get().normalize(2)));
    }

  }
}
