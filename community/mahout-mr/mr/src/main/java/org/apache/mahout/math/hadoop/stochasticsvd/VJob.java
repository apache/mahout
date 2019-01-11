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
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.PlusMult;

public class VJob {
  private static final String OUTPUT_V = "v";
  private static final String PROP_UHAT_PATH = "ssvd.uhat.path";
  private static final String PROP_SIGMA_PATH = "ssvd.sigma.path";
  private static final String PROP_OUTPUT_SCALING = "ssvd.v.output.scaling";
  private static final String PROP_K = "ssvd.k";
  public static final String PROP_SQ_PATH = "ssvdpca.sq.path";
  public static final String PROP_XI_PATH = "ssvdpca.xi.path";

  private Job job;

  public static final class VMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

    private Matrix uHat;
    private Vector vRow;
    private Vector sValues;
    private VectorWritable vRowWritable;
    private int kp;
    private int k;
    /*
     * xi and s_q are PCA-related corrections, per MAHOUT-817
     */
    private Vector xi;
    private Vector sq;
    private final PlusMult plusMult = new PlusMult(0);

    @Override
    protected void map(IntWritable key, VectorWritable value, Context context)
      throws IOException, InterruptedException {
      Vector bCol = value.get();
      /*
       * MAHOUT-817: PCA correction for B': b_{col=i} -= s_q * xi_{i}
       */
      if (xi != null) {
        /*
         * code defensively against shortened xi which may be externally
         * supplied
         */
        int btIndex = key.get();
        double xii = xi.size() > btIndex ? xi.getQuick(btIndex) : 0.0;
        plusMult.setMultiplicator(-xii);
        bCol.assign(sq, plusMult);
      }

      for (int i = 0; i < k; i++) {
        vRow.setQuick(i, bCol.dot(uHat.viewColumn(i)) / sValues.getQuick(i));
      }
      context.write(key, vRowWritable);
    }

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {
      super.setup(context);

      Configuration conf = context.getConfiguration();
      FileSystem fs = FileSystem.get(conf);
      Path uHatPath = new Path(conf.get(PROP_UHAT_PATH));

      Path sigmaPath = new Path(conf.get(PROP_SIGMA_PATH));

      uHat = SSVDHelper.drmLoadAsDense(fs, uHatPath, conf);
      // since uHat is (k+p) x (k+p)
      kp = uHat.columnSize();
      k = context.getConfiguration().getInt(PROP_K, kp);
      vRow = new DenseVector(k);
      vRowWritable = new VectorWritable(vRow);

      sValues = SSVDHelper.loadVector(sigmaPath, conf);
      SSVDSolver.OutputScalingEnum outputScaling =
        SSVDSolver.OutputScalingEnum.valueOf(context.getConfiguration()
                                                    .get(PROP_OUTPUT_SCALING));
      switch (outputScaling) {
        case SIGMA:
          sValues.assign(1.0);
          break;
        case HALFSIGMA:
          sValues = SSVDHelper.loadVector(sigmaPath, context.getConfiguration());
          sValues.assign(Functions.SQRT);
          break;
        default:
      }

      /*
       * PCA -related corrections (MAHOUT-817)
       */
      String xiPathStr = conf.get(PROP_XI_PATH);
      if (xiPathStr != null) {
        xi = SSVDHelper.loadAndSumUpVectors(new Path(xiPathStr), conf);
        sq =
          SSVDHelper.loadAndSumUpVectors(new Path(conf.get(PROP_SQ_PATH)), conf);
      }

    }

  }

  /**
   * 
   * @param conf
   * @param inputPathBt
   * @param xiPath
   *          PCA row mean (MAHOUT-817, to fix B')
   * @param sqPath
   *          sq (MAHOUT-817, to fix B')
   * @param inputUHatPath
   * @param inputSigmaPath
   * @param outputPath
   * @param k
   * @param numReduceTasks
   * @param outputScaling output scaling: apply Sigma, or Sigma^0.5, or none
   * @throws ClassNotFoundException
   * @throws InterruptedException
   * @throws IOException
   */
  public void run(Configuration conf,
                  Path inputPathBt,
                  Path xiPath,
                  Path sqPath,

                  Path inputUHatPath,
                  Path inputSigmaPath,

                  Path outputPath,
                  int k,
                  int numReduceTasks,
                  SSVDSolver.OutputScalingEnum outputScaling) throws ClassNotFoundException,
    InterruptedException, IOException {

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
    SequenceFileOutputFormat.setOutputCompressionType(job,
                                                      CompressionType.BLOCK);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(VMapper.class);

    job.getConfiguration().set(PROP_UHAT_PATH, inputUHatPath.toString());
    job.getConfiguration().set(PROP_SIGMA_PATH, inputSigmaPath.toString());
    job.getConfiguration().set(PROP_OUTPUT_SCALING, outputScaling.name());
    job.getConfiguration().setInt(PROP_K, k);
    job.setNumReduceTasks(0);

    /*
     * PCA-related options, MAHOUT-817
     */
    if (xiPath != null) {
      job.getConfiguration().set(PROP_XI_PATH, xiPath.toString());
      job.getConfiguration().set(PROP_SQ_PATH, sqPath.toString());
    }

    job.submit();

  }

  public void waitForCompletion() throws IOException, ClassNotFoundException,
    InterruptedException {
    job.waitForCompletion(false);

    if (!job.isSuccessful()) {
      throw new IOException("V job unsuccessful.");
    }

  }

}
