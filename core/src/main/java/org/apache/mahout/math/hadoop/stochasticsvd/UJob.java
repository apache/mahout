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
import org.apache.hadoop.io.Writable;
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
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

/**
 * Computes U=Q*Uhat of SSVD (optionally adding x pow(Sigma, 0.5) )
 * 
 */
public class UJob {
  private static final String OUTPUT_U = "u";
  private static final String PROP_UHAT_PATH = "ssvd.uhat.path";
  private static final String PROP_SIGMA_PATH = "ssvd.sigma.path";
  private static final String PROP_OUTPUT_SCALING = "ssvd.u.output.scaling";
  private static final String PROP_K = "ssvd.k";

  private Job job;

  public void run(Configuration conf, Path inputPathQ, Path inputUHatPath,
      Path sigmaPath, Path outputPath, int k, int numReduceTasks,
      Class<? extends Writable> labelClass, SSVDSolver.OutputScalingEnum outputScaling)
    throws ClassNotFoundException, InterruptedException, IOException {

    job = new Job(conf);
    job.setJobName("U-job");
    job.setJarByClass(UJob.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.setInputPaths(job, inputPathQ);
    FileOutputFormat.setOutputPath(job, outputPath);

    // WARN: tight hadoop integration here:
    job.getConfiguration().set("mapreduce.output.basename", OUTPUT_U);
    FileOutputFormat.setCompressOutput(job, true);
    FileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);

    job.setMapperClass(UMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    job.setOutputKeyClass(labelClass);
    job.setOutputValueClass(VectorWritable.class);

    job.getConfiguration().set(PROP_UHAT_PATH, inputUHatPath.toString());
    job.getConfiguration().set(PROP_SIGMA_PATH, sigmaPath.toString());
    job.getConfiguration().set(PROP_OUTPUT_SCALING, outputScaling.name());
    job.getConfiguration().setInt(PROP_K, k);
    job.setNumReduceTasks(0);
    job.submit();

  }

  public void waitForCompletion() throws IOException, ClassNotFoundException,
      InterruptedException {
    job.waitForCompletion(false);

    if (!job.isSuccessful()) {
      throw new IOException("U job unsuccessful.");
    }

  }

  public static final class UMapper extends
      Mapper<Writable, VectorWritable, Writable, VectorWritable> {

    private Matrix uHat;
    private DenseVector uRow;
    private VectorWritable uRowWritable;
    private int kp;
    private int k;
    private Vector sValues;

    @Override
    protected void map(Writable key, VectorWritable value, Context context)
      throws IOException, InterruptedException {
      Vector qRow = value.get();
      if (sValues != null) {
        for (int i = 0; i < k; i++) {
          uRow.setQuick(i,
                        qRow.dot(uHat.viewColumn(i)) * sValues.getQuick(i));
        }
      } else {
        for (int i = 0; i < k; i++) {
          uRow.setQuick(i, qRow.dot(uHat.viewColumn(i)));
        }
      }

      /*
       * MAHOUT-1067: inherit A names too.
       */
      if (qRow instanceof NamedVector) {
        uRowWritable.set(new NamedVector(uRow, ((NamedVector) qRow).getName()));
      } else {
        uRowWritable.set(uRow);
      }

      context.write(key, uRowWritable); // U inherits original A row labels.
    }

    @Override
    protected void setup(Context context) throws IOException,
        InterruptedException {
      super.setup(context);
      Path uHatPath = new Path(context.getConfiguration().get(PROP_UHAT_PATH));
      Path sigmaPath = new Path(context.getConfiguration().get(PROP_SIGMA_PATH));
      FileSystem fs = FileSystem.get(uHatPath.toUri(), context.getConfiguration());

      uHat = SSVDHelper.drmLoadAsDense(fs, uHatPath, context.getConfiguration());
      // since uHat is (k+p) x (k+p)
      kp = uHat.columnSize();
      k = context.getConfiguration().getInt(PROP_K, kp);
      uRow = new DenseVector(k);
      uRowWritable = new VectorWritable(uRow);

      SSVDSolver.OutputScalingEnum outputScaling =
        SSVDSolver.OutputScalingEnum.valueOf(context.getConfiguration()
                                                    .get(PROP_OUTPUT_SCALING));
      switch (outputScaling) {
        case SIGMA:
          sValues = SSVDHelper.loadVector(sigmaPath, context.getConfiguration());
          break;
        case HALFSIGMA:
          sValues = SSVDHelper.loadVector(sigmaPath, context.getConfiguration());
          sValues.assign(Functions.SQRT);
          break;
        default:
      }
    }

  }

}
