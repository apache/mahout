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

package org.apache.mahout.math.hadoop.stochasticsvd;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class VJob {
  private static final String OUTPUT_V = "v";
  private static final String PROP_UHAT_PATH = "ssvd.uhat.path";
  private static final String PROP_SIGMA_PATH = "ssvd.sigma.path";
  private static final String PROP_V_HALFSIGMA = "ssvd.v.halfsigma";
  private static final String PROP_K = "ssvd.k";

  private Job job;

  public void start(Configuration conf, Path inputPathBt, Path inputUHatPath,
      Path inputSigmaPath, Path outputPath, int k, int numReduceTasks,
      boolean vHalfSigma) throws ClassNotFoundException, InterruptedException,
      IOException {

    job = new Job(conf);
    job.setJobName("V-job");
    job.setJarByClass(VJob.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.setInputPaths(job, inputPathBt);
    FileOutputFormat.setOutputPath(job, outputPath);

    // Warn: tight hadoop integration here:
    job.getConfiguration().set("mapreduce.output.basename", OUTPUT_V);
    FileOutputFormat.setCompressOutput(job, true);
    FileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(VMapper.class);

    job.getConfiguration().set(PROP_UHAT_PATH, inputUHatPath.toString());
    job.getConfiguration().set(PROP_SIGMA_PATH, inputSigmaPath.toString());
    if (vHalfSigma) {
      job.getConfiguration().set(PROP_V_HALFSIGMA, "y");
    }
    job.getConfiguration().setInt(PROP_K, k);
    job.setNumReduceTasks(0);
    job.submit();

  }

  public void waitForCompletion() throws IOException, ClassNotFoundException,
      InterruptedException {
    job.waitForCompletion(false);

    if (!job.isSuccessful()) {
      throw new IOException("V job unsuccessful.");
    }

  }

  public static final class VMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

    private Matrix uHat;
    private DenseVector vRow;
    private DenseVector sValues;
    private VectorWritable vRowWritable;
    private int kp;
    private int k;

    @Override
    protected void map(IntWritable key, VectorWritable value, Context context)
      throws IOException, InterruptedException {
      Vector qRow = value.get();
      for (int i = 0; i < k; i++) {
        vRow.setQuick(i,
                      qRow.dot(uHat.viewColumn(i)) / sValues.getQuick(i));
      }
      context.write(key, vRowWritable); // U inherits original A row labels.
    }

    @Override
    protected void setup(Context context) throws IOException,
        InterruptedException {
      super.setup(context);
      FileSystem fs = FileSystem.get(context.getConfiguration());
      Path uHatPath = new Path(context.getConfiguration().get(PROP_UHAT_PATH));

      Path sigmaPath = new Path(context.getConfiguration().get(PROP_SIGMA_PATH));

      uHat = new DenseMatrix(SSVDSolver.loadDistributedRowMatrix(fs,
          uHatPath, context.getConfiguration()));
      // since uHat is (k+p) x (k+p)
      kp = uHat.columnSize();
      k = context.getConfiguration().getInt(PROP_K, kp);
      vRow = new DenseVector(k);
      vRowWritable = new VectorWritable(vRow);

      sValues = new DenseVector(SSVDSolver.loadDistributedRowMatrix(fs,
          sigmaPath, context.getConfiguration())[0], true);
      if (context.getConfiguration().get(PROP_V_HALFSIGMA) != null) {
        for (int i = 0; i < k; i++) {
          sValues.setQuick(i, Math.sqrt(sValues.getQuick(i)));
        }
      }

    }

  }

}
