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
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Given a matrix, this job returns a vector whose i_th element is the 
 * sum of all the elements in the i_th row of the original matrix.
 */
public final class MatrixDiagonalizeJob {

  private MatrixDiagonalizeJob() {
  }

  public static Vector runJob(Path affInput, int dimensions)
    throws IOException, ClassNotFoundException, InterruptedException {
    
    // set up all the job tasks
    Configuration conf = new Configuration();
    Path diagOutput = new Path(affInput.getParent(), "diagonal");
    HadoopUtil.delete(conf, diagOutput);
    conf.setInt(Keys.AFFINITY_DIMENSIONS, dimensions);
    Job job = new Job(conf, "MatrixDiagonalizeJob");
    
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapOutputKeyClass(NullWritable.class);
    job.setMapOutputValueClass(IntDoublePairWritable.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(MatrixDiagonalizeMapper.class);
    job.setReducerClass(MatrixDiagonalizeReducer.class);
    
    FileInputFormat.addInputPath(job, affInput);
    FileOutputFormat.setOutputPath(job, diagOutput);
    
    job.setJarByClass(MatrixDiagonalizeJob.class);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }

    // read the results back from the path
    return VectorCache.load(conf, new Path(diagOutput, "part-r-00000"));
  }
  
  public static class MatrixDiagonalizeMapper
    extends Mapper<IntWritable, VectorWritable, NullWritable, IntDoublePairWritable> {
    
    @Override
    protected void map(IntWritable key, VectorWritable row, Context context) 
      throws IOException, InterruptedException {
      // store the sum
      IntDoublePairWritable store = new IntDoublePairWritable(key.get(), row.get().zSum());
      context.write(NullWritable.get(), store);
    }
  }
  
  public static class MatrixDiagonalizeReducer
    extends Reducer<NullWritable, IntDoublePairWritable, NullWritable, VectorWritable> {
    
    @Override
    protected void reduce(NullWritable key, Iterable<IntDoublePairWritable> values,
      Context context) throws IOException, InterruptedException {
      // create the return vector
      Vector retval = new DenseVector(context.getConfiguration().getInt(Keys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE));
      // put everything in its correct spot
      for (IntDoublePairWritable e : values) {
        retval.setQuick(e.getKey(), e.getValue());
      }
      // write it out
      context.write(key, new VectorWritable(retval));
    }
  }
}
