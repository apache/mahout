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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.QJob.QJobKeyWritable;

/**
 * Bt job. For details, see working notes in MAHOUT-376. 
 *
 */
@SuppressWarnings("deprecation")
public class BtJob {

  public static final String OUTPUT_Q = "Q";
  public static final String OUTPUT_BT = "part";
  public static final String PROP_QJOB_PATH = "ssvd.QJob.path";

  public static class BtMapper extends
      Mapper<Writable, VectorWritable, IntWritable, VectorWritable> {

    private SequenceFile.Reader qInput;
    private List<UpperTriangular> mRs = new ArrayList<UpperTriangular>();
    private int blockNum;
    private double[][] mQt;
    private int cnt;
    private int r;
    private MultipleOutputs outputs;
    private IntWritable btKey = new IntWritable();
    private VectorWritable btValue = new VectorWritable();
    private int kp;
    private VectorWritable qRowValue = new VectorWritable();
    private int qCount; // debug

    void loadNextQt(Context ctx) throws IOException, InterruptedException {
      QJobKeyWritable key = new QJobKeyWritable();
      DenseBlockWritable v = new DenseBlockWritable();

      boolean more = qInput.next(key, v);
      assert more;

      mQt = GivensThinSolver.computeQtHat(v.getBlock(), blockNum == 0 ? 0
          : 1, new GivensThinSolver.DeepCopyUTIterator(mRs.iterator()));
      r = mQt[0].length;
      kp = mQt.length;
      if (btValue.get() == null)
        btValue.set(new DenseVector(kp));
      if (qRowValue.get() == null)
        qRowValue.set(new DenseVector(kp));

      // also output QHat -- although we don't know the A labels there. Is it
      // important?
      // DenseVector qRow = new DenseVector(m_kp);
      // IntWritable oKey = new IntWritable();
      // VectorWritable oV = new VectorWritable();
      //
      // for ( int i = m_r-1; i>=0; i-- ) {
      // for ( int j= 0; j < m_kp; j++ )
      // qRow.setQuick(j, m_qt[j][i]);
      // oKey.set((m_blockNum<<20)+m_r-i-1);
      // oV.set(qRow);
      // // so the block #s range is thus 0..2048, and number of rows per block
      // is 0..2^20.
      // // since we are not really sending it out to sort (it is a 'side
      // file'),
      // // it doesn't matter if it overflows.
      // m_outputs.write( OUTPUT_Q, oKey, oV);
      // }
      qCount++;
    }

    @Override
    protected void cleanup(Context context) throws IOException,
        InterruptedException {

      if (qInput != null)
        qInput.close();
      if (outputs != null)
        outputs.close();
      super.cleanup(context);
    }

    @Override
    @SuppressWarnings ({"unchecked"})
    protected void map(Writable key, VectorWritable value, Context context)
        throws IOException, InterruptedException {
      if (mQt != null && cnt++ == r)
        mQt = null;
      if (mQt == null) {
        loadNextQt(context);
        cnt = 1;
      }

      // output Bt outer products
      Vector aRow = value.get();
      int qRowIndex = r - cnt; // because QHats are initially stored in
                                   // reverse
      Vector qRow = qRowValue.get();
      for (int j = 0; j < kp; j++)
        qRow.setQuick(j, mQt[j][qRowIndex]);

      outputs.getCollector(OUTPUT_Q, null).collect(key, qRowValue); // make
                                                                        // sure
                                                                        // Qs
                                                                        // are
                                                                        // inheriting
                                                                        // A row
                                                                        // labels.

      int n = aRow.size();
      Vector btRow = btValue.get();
      for (int i = 0; i < n; i++) {
        double mul = aRow.getQuick(i);
        for (int j = 0; j < kp; j++)
          btRow.setQuick(j, mul * qRow.getQuick(j));
        btKey.set(i);
        context.write(btKey, btValue);
      }

    }

    @Override
    protected void setup(Context context) throws IOException,
        InterruptedException {
      super.setup(context);

      Path qJobPath = new Path(context.getConfiguration().get(PROP_QJOB_PATH));

      FileSystem fs = FileSystem.get(context.getConfiguration());
      // actually this is kind of dangerous
      // becuase this routine thinks we need to create file name for
      // our current job and this will use -m- so it's just serendipity we are
      // calling
      // it from the mapper too as the QJob did.
      Path qInputPath = new Path(qJobPath, FileOutputFormat.getUniqueFile(
          context, QJob.OUTPUT_QHAT, ""));
      qInput = new SequenceFile.Reader(fs, qInputPath,
          context.getConfiguration());

      blockNum = context.getTaskAttemptID().getTaskID().getId();

      // read all r files _in order of task ids_, i.e. partitions
      Path rPath = new Path(qJobPath, QJob.OUTPUT_R + "-*");
      FileStatus[] rFiles = fs.globStatus(rPath);

      if (rFiles == null)
        throw new IOException("Can't find R inputs ");

      Arrays.sort(rFiles, SSVDSolver.partitionComparator);

      QJobKeyWritable rKey = new QJobKeyWritable();
      VectorWritable rValue = new VectorWritable();

      int block = 0;
      for (FileStatus fstat : rFiles) {
        SequenceFile.Reader rReader = new SequenceFile.Reader(fs,
            fstat.getPath(), context.getConfiguration());
        try {
          rReader.next(rKey, rValue);
        } finally {
          rReader.close();
        }
        if (block < blockNum && block > 0)
          GivensThinSolver.mergeR(mRs.get(0),
              new UpperTriangular(rValue.get()));
        else
          mRs.add(new UpperTriangular(rValue.get()));
        block++;
      }
      outputs = new MultipleOutputs(new JobConf(context.getConfiguration()));
    }
  }

  public static class OuterProductReducer extends
      Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

    private VectorWritable oValue = new VectorWritable();
    private DenseVector accum;

    @Override
    protected void reduce(IntWritable key, Iterable<VectorWritable> values,
        Context ctx) throws IOException, InterruptedException {
      Iterator<VectorWritable> vwIter = values.iterator();

      Vector vec = vwIter.next().get();
      if (accum == null || accum.size() != vec.size()) {
        accum = new DenseVector(vec);
        oValue.set(accum);
      } else
        accum.assign(vec);

      while (vwIter.hasNext())
        accum.addAll(vwIter.next().get());
      ctx.write(key, oValue);
    }

  }

  public static void run(Configuration conf, Path inputPathA[],
      Path inputPathQJob, Path outputPath, int minSplitSize, int k, int p,
      int numReduceTasks, Class<? extends Writable> labelClass)
      throws ClassNotFoundException, InterruptedException, IOException {

    JobConf oldApiJob = new JobConf(conf);
    MultipleOutputs.addNamedOutput(oldApiJob, OUTPUT_Q,
        org.apache.hadoop.mapred.SequenceFileOutputFormat.class, labelClass,
        VectorWritable.class);

    Job job = new Job(oldApiJob);
    job.setJobName("Bt-job");
    job.setJarByClass(BtJob.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.setInputPaths(job, inputPathA);
    if (minSplitSize > 0)
      SequenceFileInputFormat.setMinInputSplitSize(job, minSplitSize);
    FileOutputFormat.setOutputPath(job, outputPath);

    // MultipleOutputs.addNamedOutput(job, OUTPUT_Bt,
    // SequenceFileOutputFormat.class,
    // QJobKeyWritable.class,QJobValueWritable.class);

    // Warn: tight hadoop integration here:
    job.getConfiguration().set("mapreduce.output.basename", OUTPUT_BT);
    SequenceFileOutputFormat.setCompressOutput(job, true);
    SequenceFileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(job,
        CompressionType.BLOCK);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(BtMapper.class);
    job.setCombinerClass(OuterProductReducer.class);
    job.setReducerClass(OuterProductReducer.class);
    // job.setPartitionerClass(QPartitioner.class);

    // job.getConfiguration().setInt(QJob.PROP_AROWBLOCK_SIZE,aBlockRows );
    // job.getConfiguration().setLong(PROP_OMEGA_SEED, seed);
    job.getConfiguration().setInt(QJob.PROP_K, k);
    job.getConfiguration().setInt(QJob.PROP_P, p);
    job.getConfiguration().set(PROP_QJOB_PATH, inputPathQJob.toString());

    // number of reduce tasks doesn't matter. we don't actually
    // send anything to reducers. in fact, the only reason
    // we need to configure reduce step is so that combiners can fire.
    // so reduce here is purely symbolic.
    job.setNumReduceTasks(numReduceTasks);

    job.submit();
    job.waitForCompletion(false);

    if (!job.isSuccessful())
      throw new IOException("Bt job unsuccessful.");

  }

}
