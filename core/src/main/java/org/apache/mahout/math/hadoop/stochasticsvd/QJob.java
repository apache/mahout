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
import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.iterator.CopyConstructorIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;

/**
 * Compute first level of QHat-transpose blocks.
 * 
 * See Mahout-376 woking notes for details.
 * 
 * 
 */
public final class QJob {

  public static final String PROP_OMEGA_SEED = "ssvd.omegaseed";
  public static final String PROP_K = "ssvd.k";
  public static final String PROP_P = "ssvd.p";
  public static final String PROP_AROWBLOCK_SIZE = "ssvd.arowblock.size";

  public static final String OUTPUT_R = "R";
  public static final String OUTPUT_QHAT = "QHat";

  private QJob() {
  }
  // public static final String OUTPUT_Q="Q";
  //public static final String OUTPUT_BT = "Bt";

  public static class QJobKeyWritable implements WritableComparable<QJobKeyWritable> {

    private int taskId;
    private int taskRowOrdinal;

    @Override
    public void readFields(DataInput in) throws IOException {
      taskId = in.readInt();
      taskRowOrdinal = in.readInt();
    }

    @Override
    public void write(DataOutput out) throws IOException {
      out.writeInt(taskId);
      out.writeInt(taskRowOrdinal);
    }

    @Override
    public int compareTo(QJobKeyWritable o) {
      if (taskId < o.taskId) {
        return -1;
      } else if (taskId > o.taskId) {
        return 1;
      }
      if (taskRowOrdinal < o.taskRowOrdinal) {
        return -1;
      } else if (taskRowOrdinal > o.taskRowOrdinal) {
        return 1;
      }
      return 0;
    }

  }

  public static class QMapper extends Mapper<Writable, VectorWritable, QJobKeyWritable, VectorWritable> {

    private int kp;
    private Omega omega;
    private List<double[]> yLookahead;
    private GivensThinSolver qSolver;
    private int blockCnt;
    // private int m_reducerCount;
    private int r;
    private final DenseBlockWritable value = new DenseBlockWritable();
    private final QJobKeyWritable key = new QJobKeyWritable();
    private final Writable tempKey = new IntWritable();
    private MultipleOutputs outputs;
    private final Deque<Closeable> closeables = new LinkedList<Closeable>();
    private SequenceFile.Writer tempQw;
    private Path tempQPath;
    private final List<UpperTriangular> rSubseq = new ArrayList<UpperTriangular>();

    private void flushSolver(Context context) throws IOException {
      UpperTriangular r = qSolver.getRTilde();
      double[][] qt = qSolver.getThinQtTilde();

      rSubseq.add(r);

      value.setBlock(qt);
      getTempQw(context).append(tempKey, value); // this probably should be
                                                     // a sparse row matrix,
      // but compressor should get it for disk and in memory we want it
      // dense anyway, sparse random implementations would be
      // a mostly a memory management disaster consisting of rehashes and GC
      // thrashing. (IMHO)
      value.setBlock(null);
      qSolver.reset();
    }

    // second pass to run a modified version of computeQHatSequence.
    private void flushQBlocks(Context ctx) throws IOException {
      if (blockCnt == 1) {
        // only one block, no temp file, no second pass. should be the default
        // mode
        // for efficiency in most cases. Sure mapper should be able to load
        // the entire split in memory -- and we don't require even that.
        value.setBlock(qSolver.getThinQtTilde());
        outputs.getCollector(OUTPUT_QHAT, null).collect(key, value);
        outputs.getCollector(OUTPUT_R, null).collect(
            key,
            new VectorWritable(new DenseVector(qSolver.getRTilde().getData(),
                true)));

      } else {
        secondPass(ctx);
      }
    }

    private void secondPass(Context ctx) throws IOException {
      qSolver = null; // release mem
      FileSystem localFs = FileSystem.getLocal(ctx.getConfiguration());
      SequenceFile.Reader tempQr = new SequenceFile.Reader(localFs, tempQPath, ctx.getConfiguration());
      closeables.addFirst(tempQr);
      int qCnt = 0;
      while (tempQr.next(tempKey, value)) {
        value.setBlock(GivensThinSolver.computeQtHat(value.getBlock(),
                                                     qCnt,
                                                     new CopyConstructorIterator<UpperTriangular>(rSubseq.iterator())));
        if (qCnt == 1) {
          // just merge r[0] <- r[1] so it doesn't have to repeat
          // in subsequent computeQHat iterators
          GivensThinSolver.mergeR(rSubseq.get(0), rSubseq.remove(1));
        } else {
          qCnt++;
        }
        outputs.getCollector(OUTPUT_QHAT, null).collect(key, value);
      }

      assert rSubseq.size() == 1;

      // m_value.setR(m_rSubseq.get(0));
      outputs.getCollector(OUTPUT_R, null).collect(
          key,
          new VectorWritable(new DenseVector(rSubseq.get(0).getData(),
                                             true)));

    }

    @Override
    protected void map(Writable key, VectorWritable value, Context context)
      throws IOException, InterruptedException {
      double[] yRow;
      if (yLookahead.size() == kp) {
        if (qSolver.isFull()) {

          flushSolver(context);
          blockCnt++;

        }
        yRow = yLookahead.remove(0);

        qSolver.appendRow(yRow);
      } else {
        yRow = new double[kp];
      }
      omega.computeYRow(value.get(), yRow);
      yLookahead.add(yRow);
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {

      int k = Integer.parseInt(context.getConfiguration().get(PROP_K));
      int p = Integer.parseInt(context.getConfiguration().get(PROP_P));
      kp = k + p;
      long omegaSeed = Long.parseLong(context.getConfiguration().get(PROP_OMEGA_SEED));
      r = Integer.parseInt(context.getConfiguration().get(PROP_AROWBLOCK_SIZE));
      omega = new Omega(omegaSeed, k, p);
      yLookahead = new ArrayList<double[]>(kp);
      qSolver = new GivensThinSolver(r, kp);
      outputs = new MultipleOutputs(new JobConf(context.getConfiguration()));
      closeables.addFirst(new Closeable() {
        @Override
        public void close() throws IOException {
          outputs.close();
        }
      });

    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      try {
        if (qSolver == null && yLookahead.isEmpty()) {
          return;
        }
        if (qSolver == null) {
          qSolver = new GivensThinSolver(yLookahead.size(), kp);
        }
        // grow q solver up if necessary

        qSolver.adjust(qSolver.getCnt() + yLookahead.size());
        while (!yLookahead.isEmpty()) {

          qSolver.appendRow(yLookahead.remove(0));

        }
        assert qSolver.isFull();
        if (++blockCnt > 1) {
          flushSolver(context);
          assert tempQw != null;
          closeables.remove(tempQw);
          tempQw.close();
        }
        flushQBlocks(context);

      } finally {
        IOUtils.close(closeables);
      }

    }

    private SequenceFile.Writer getTempQw(Context context) throws IOException {
      if (tempQw == null) {
        // temporary Q output
        // hopefully will not exceed size of IO cache in which case it is only
        // good since it
        // is going to be maanged by kernel, not java GC. And if IO cache is not
        // good enough,
        // then at least it is always sequential.
        String taskTmpDir = System.getProperty("java.io.tmpdir");
        FileSystem localFs = FileSystem.getLocal(context.getConfiguration());
        tempQPath = new Path(new Path(taskTmpDir), "q-temp.seq");
        tempQw = SequenceFile.createWriter(localFs,
            context.getConfiguration(), tempQPath, IntWritable.class,
            DenseBlockWritable.class, CompressionType.BLOCK);
        closeables.addFirst(tempQw);
        closeables.addFirst(new IOUtils.DeleteFileOnClose(new File(tempQPath.toString())));
      }
      return tempQw;
    }
  }

  public static void run(Configuration conf,
                         Path[] inputPaths,
                         Path outputPath,
                         int aBlockRows,
                         int minSplitSize,
                         int k,
                         int p,
                         long seed,
                         int numReduceTasks) throws ClassNotFoundException, InterruptedException, IOException {

    JobConf oldApiJob = new JobConf(conf);
    MultipleOutputs.addNamedOutput(oldApiJob, OUTPUT_QHAT,
        org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
        QJobKeyWritable.class, DenseBlockWritable.class);
    MultipleOutputs.addNamedOutput(oldApiJob, OUTPUT_R,
        org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
        QJobKeyWritable.class, VectorWritable.class);

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

    job.setMapOutputKeyClass(QJobKeyWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    job.setOutputKeyClass(QJobKeyWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(QMapper.class);

    job.getConfiguration().setInt(PROP_AROWBLOCK_SIZE, aBlockRows);
    job.getConfiguration().setLong(PROP_OMEGA_SEED, seed);
    job.getConfiguration().setInt(PROP_K, k);
    job.getConfiguration().setInt(PROP_P, p);

    // number of reduce tasks doesn't matter. we don't actually
    // send anything to reducers. in fact, the only reason
    // we need to configure reduce step is so that combiners can fire.
    // so reduce here is purely symbolic.
    job.setNumReduceTasks(0 /* numReduceTasks */);

    job.submit();
    job.waitForCompletion(false);

    if (!job.isSuccessful()) {
      throw new IOException("Q job unsuccessful.");
    }

  }

}
