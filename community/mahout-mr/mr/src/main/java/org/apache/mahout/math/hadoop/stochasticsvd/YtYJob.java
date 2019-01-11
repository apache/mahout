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

import org.apache.commons.lang3.Validate;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.UpperTriangular;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

/**
 * Job that accumulates Y'Y output
 */
public final class YtYJob {

  public static final String PROP_OMEGA_SEED = "ssvd.omegaseed";
  public static final String PROP_K = "ssvd.k";
  public static final String PROP_P = "ssvd.p";

  // we have single output, so we use standard output
  public static final String OUTPUT_YT_Y = "part-";

  private YtYJob() {
  }

  public static class YtYMapper extends
    Mapper<Writable, VectorWritable, IntWritable, VectorWritable> {

    private int kp;
    private Omega omega;
    private UpperTriangular mYtY;

    /*
     * we keep yRow in a dense form here but keep an eye not to dense up while
     * doing YtY products. I am not sure that sparse vector would create much
     * performance benefits since we must to assume that y would be more often
     * dense than sparse, so for bulk dense operations that would perform
     * somewhat better than a RandomAccessSparse vector frequent updates.
     */
    private Vector yRow;

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {
      int k = context.getConfiguration().getInt(PROP_K, -1);
      int p = context.getConfiguration().getInt(PROP_P, -1);

      Validate.isTrue(k > 0, "invalid k parameter");
      Validate.isTrue(p > 0, "invalid p parameter");

      kp = k + p;
      long omegaSeed =
        Long.parseLong(context.getConfiguration().get(PROP_OMEGA_SEED));

      omega = new Omega(omegaSeed, k + p);

      mYtY = new UpperTriangular(kp);

      // see which one works better!
      // yRow = new RandomAccessSparseVector(kp);
      yRow = new DenseVector(kp);
    }

    @Override
    protected void map(Writable key, VectorWritable value, Context context)
      throws IOException, InterruptedException {
      omega.computeYRow(value.get(), yRow);
      // compute outer product update for YtY

      if (yRow.isDense()) {
        for (int i = 0; i < kp; i++) {
          double yi;
          if ((yi = yRow.getQuick(i)) == 0.0) {
            continue; // avoid densing up here unnecessarily
          }
          for (int j = i; j < kp; j++) {
            double yj;
            if ((yj = yRow.getQuick(j)) != 0.0) {
              mYtY.setQuick(i, j, mYtY.getQuick(i, j) + yi * yj);
            }
          }
        }
      } else {
        /*
         * the disadvantage of using sparse vector (aside from the fact that we
         * are creating some short-lived references) here is that we obviously
         * do two times more iterations then necessary if y row is pretty dense.
         */
        for (Vector.Element eli : yRow.nonZeroes()) {
          int i = eli.index();
          for (Vector.Element elj : yRow.nonZeroes()) {
            int j = elj.index();
            if (j < i) {
              continue;
            }
            mYtY.setQuick(i, j, mYtY.getQuick(i, j) + eli.get() * elj.get());
          }
        }
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException,
      InterruptedException {
      context.write(new IntWritable(context.getTaskAttemptID().getTaskID()
                                      .getId()),
                    new VectorWritable(new DenseVector(mYtY.getData())));
    }
  }

  public static class YtYReducer extends
    Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private final VectorWritable accum = new VectorWritable();
    private DenseVector acc;

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {
      int k = context.getConfiguration().getInt(PROP_K, -1);
      int p = context.getConfiguration().getInt(PROP_P, -1);

      Validate.isTrue(k > 0, "invalid k parameter");
      Validate.isTrue(p > 0, "invalid p parameter");
      accum.set(acc = new DenseVector(k + p));
    }

    @Override
    protected void cleanup(Context context) throws IOException,
      InterruptedException {
      context.write(new IntWritable(), accum);
    }

    @Override
    protected void reduce(IntWritable key,
                          Iterable<VectorWritable> values,
                          Context arg2) throws IOException,
      InterruptedException {
      for (VectorWritable vw : values) {
        acc.addAll(vw.get());
      }
    }
  }

  public static void run(Configuration conf,
                         Path[] inputPaths,
                         Path outputPath,
                         int k,
                         int p,
                         long seed) throws ClassNotFoundException,
    InterruptedException, IOException {

    Job job = new Job(conf);
    job.setJobName("YtY-job");
    job.setJarByClass(YtYJob.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.setInputPaths(job, inputPaths);
    FileOutputFormat.setOutputPath(job, outputPath);

    SequenceFileOutputFormat.setOutputCompressionType(job,
                                                      CompressionType.BLOCK);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(YtYMapper.class);

    job.getConfiguration().setLong(PROP_OMEGA_SEED, seed);
    job.getConfiguration().setInt(PROP_K, k);
    job.getConfiguration().setInt(PROP_P, p);

    /*
     * we must reduce to just one matrix which means we need only one reducer.
     * But it's ok since each mapper outputs only one vector (a packed
     * UpperTriangular) so even if there're thousands of mappers, one reducer
     * should cope just fine.
     */
    job.setNumReduceTasks(1);

    job.submit();
    job.waitForCompletion(false);

    if (!job.isSuccessful()) {
      throw new IOException("YtY job unsuccessful.");
    }

  }

}
