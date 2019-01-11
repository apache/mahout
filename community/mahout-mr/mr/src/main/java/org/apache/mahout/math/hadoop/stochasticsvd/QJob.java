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

import java.io.Closeable;
import java.io.IOException;
import java.util.Deque;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.stochasticsvd.qr.QRFirstStep;

/**
 * Compute first level of QHat-transpose blocks.
 * <P>
 * 
 * See Mahout-376 working notes for details.
 * <P>
 * 
 * Uses some of Hadoop deprecated api wherever newer api is not available.
 * Hence, @SuppressWarnings("deprecation") for imports (MAHOUT-593).
 * <P>
 * 
 */
@SuppressWarnings("deprecation")
public final class QJob {

  public static final String PROP_OMEGA_SEED = "ssvd.omegaseed";
  public static final String PROP_K = QRFirstStep.PROP_K;
  public static final String PROP_P = QRFirstStep.PROP_P;
  public static final String PROP_SB_PATH = "ssvdpca.sb.path";
  public static final String PROP_AROWBLOCK_SIZE =
    QRFirstStep.PROP_AROWBLOCK_SIZE;

  public static final String OUTPUT_RHAT = "R";
  public static final String OUTPUT_QHAT = "QHat";

  private QJob() {
  }

  public static class QMapper
      extends
      Mapper<Writable, VectorWritable, SplitPartitionedWritable, VectorWritable> {

    private MultipleOutputs outputs;
    private final Deque<Closeable> closeables = Lists.newLinkedList();
    private SplitPartitionedWritable qHatKey;
    private SplitPartitionedWritable rHatKey;
    private Vector yRow;
    private Vector sb;
    private Omega omega;
    private int kp;

    private QRFirstStep qr;

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {

      Configuration conf = context.getConfiguration();
      int k = Integer.parseInt(conf.get(PROP_K));
      int p = Integer.parseInt(conf.get(PROP_P));
      kp = k + p;
      long omegaSeed = Long.parseLong(conf.get(PROP_OMEGA_SEED));
      omega = new Omega(omegaSeed, k + p);

      String sbPathStr = conf.get(PROP_SB_PATH);
      if (sbPathStr != null) {
        sb = SSVDHelper.loadAndSumUpVectors(new Path(sbPathStr), conf);
        if (sb == null)
          throw new IOException(String.format("Unable to load s_omega from path %s.", sbPathStr));
      }

      outputs = new MultipleOutputs(new JobConf(conf));
      closeables.addFirst(new Closeable() {
        @Override
        public void close() throws IOException {
          outputs.close();
        }
      });

      qHatKey = new SplitPartitionedWritable(context);
      rHatKey = new SplitPartitionedWritable(context);

      OutputCollector<Writable, DenseBlockWritable> qhatCollector =
        new OutputCollector<Writable, DenseBlockWritable>() {

          @Override
          @SuppressWarnings("unchecked")
          public void collect(Writable nil, DenseBlockWritable dbw)
            throws IOException {
            outputs.getCollector(OUTPUT_QHAT, null).collect(qHatKey, dbw);
            qHatKey.incrementItemOrdinal();
          }
        };

      OutputCollector<Writable, VectorWritable> rhatCollector =
        new OutputCollector<Writable, VectorWritable>() {

          @Override
          @SuppressWarnings("unchecked")
          public void collect(Writable nil, VectorWritable rhat)
            throws IOException {
            outputs.getCollector(OUTPUT_RHAT, null).collect(rHatKey, rhat);
            rHatKey.incrementItemOrdinal();
          }
        };

      qr = new QRFirstStep(conf, qhatCollector, rhatCollector);
      closeables.addFirst(qr); // important: qr closes first!!
      yRow = new DenseVector(kp);
    }

    @Override
    protected void map(Writable key, VectorWritable value, Context context)
      throws IOException, InterruptedException {
      omega.computeYRow(value.get(), yRow);
      if (sb != null) {
        yRow.assign(sb, Functions.MINUS);
      }
      qr.collect(key, yRow);
    }

    @Override
    protected void cleanup(Context context) throws IOException,
      InterruptedException {
      IOUtils.close(closeables);
    }
  }

  public static void run(Configuration conf,
                         Path[] inputPaths,
                         Path sbPath,
                         Path outputPath,
                         int aBlockRows,
                         int minSplitSize,
                         int k,
                         int p,
                         long seed,
                         int numReduceTasks) throws ClassNotFoundException,
    InterruptedException, IOException {

    JobConf oldApiJob = new JobConf(conf);
    MultipleOutputs.addNamedOutput(oldApiJob,
                                   OUTPUT_QHAT,
                                   org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
                                   SplitPartitionedWritable.class,
                                   DenseBlockWritable.class);
    MultipleOutputs.addNamedOutput(oldApiJob,
                                   OUTPUT_RHAT,
                                   org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
                                   SplitPartitionedWritable.class,
                                   VectorWritable.class);

    Job job = new Job(oldApiJob);
    job.setJobName("Q-job");
    job.setJarByClass(QJob.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.setInputPaths(job, inputPaths);
    if (minSplitSize > 0) {
      FileInputFormat.setMinInputSplitSize(job, minSplitSize);
    }

    FileOutputFormat.setOutputPath(job, outputPath);

    FileOutputFormat.setCompressOutput(job, true);
    FileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(job,
                                                      CompressionType.BLOCK);

    job.setMapOutputKeyClass(SplitPartitionedWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    job.setOutputKeyClass(SplitPartitionedWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(QMapper.class);

    job.getConfiguration().setInt(PROP_AROWBLOCK_SIZE, aBlockRows);
    job.getConfiguration().setLong(PROP_OMEGA_SEED, seed);
    job.getConfiguration().setInt(PROP_K, k);
    job.getConfiguration().setInt(PROP_P, p);
    if (sbPath != null) {
      job.getConfiguration().set(PROP_SB_PATH, sbPath.toString());
    }

    /*
     * number of reduce tasks doesn't matter. we don't actually send anything to
     * reducers.
     */

    job.setNumReduceTasks(0 /* numReduceTasks */);

    job.submit();
    job.waitForCompletion(false);

    if (!job.isSuccessful()) {
      throw new IOException("Q job unsuccessful.");
    }

  }

}
