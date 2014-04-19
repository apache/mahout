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
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.UpperTriangular;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.PlusMult;
import org.apache.mahout.math.hadoop.stochasticsvd.qr.QRLastStep;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Bt job. For details, see working notes in MAHOUT-376.
 * <p/>
 * <p/>
 * Uses hadoop deprecated API wherever new api has not been updated
 * (MAHOUT-593), hence @SuppressWarning("deprecation").
 * <p/>
 * <p/>
 * This job outputs either Bt in its standard output, or upper triangular
 * matrices representing BBt partial sums if that's requested . If the latter
 * mode is enabled, then we accumulate BBt outer product sums in upper
 * triangular accumulator and output it at the end of the job, thus saving space
 * and BBt job.
 * <p/>
 * <p/>
 * This job also outputs Q and Bt and optionally BBt. Bt is output to standard
 * job output (part-*) and Q and BBt use named multiple outputs.
 * <p/>
 * <p/>
 */
@SuppressWarnings("deprecation")
public final class BtJob {

  public static final String OUTPUT_Q = "Q";
  public static final String OUTPUT_BT = "part";
  public static final String OUTPUT_BBT = "bbt";
  public static final String OUTPUT_SQ = "sq";
  public static final String OUTPUT_SB = "sb";

  public static final String PROP_QJOB_PATH = "ssvd.QJob.path";
  public static final String PROP_OUPTUT_BBT_PRODUCTS =
    "ssvd.BtJob.outputBBtProducts";
  public static final String PROP_OUTER_PROD_BLOCK_HEIGHT =
    "ssvd.outerProdBlockHeight";
  public static final String PROP_RHAT_BROADCAST = "ssvd.rhat.broadcast";
  public static final String PROP_XI_PATH = "ssvdpca.xi.path";
  public static final String PROP_NV = "ssvd.nv";

  static final double SPARSE_ZEROS_PCT_THRESHOLD = 0.1;

  private BtJob() {
  }

  public static class BtMapper extends
    Mapper<Writable, VectorWritable, LongWritable, SparseRowBlockWritable> {

    private QRLastStep qr;
    private final Deque<Closeable> closeables = new ArrayDeque<Closeable>();

    private int blockNum;
    private MultipleOutputs outputs;
    private final VectorWritable qRowValue = new VectorWritable();
    private Vector btRow;
    private SparseRowBlockAccumulator btCollector;
    private Context mapContext;
    private boolean nv;

    // pca stuff
    private Vector sqAccum;
    private boolean computeSq;

    /**
     * We maintain A and QtHat inputs partitioned the same way, so we
     * essentially are performing map-side merge here of A and QtHats except
     * QtHat is stored not row-wise but block-wise.
     */
    @Override
    protected void map(Writable key, VectorWritable value, Context context)
      throws IOException, InterruptedException {

      mapContext = context;
      // output Bt outer products
      Vector aRow = value.get();

      Vector qRow = qr.next();
      int kp = qRow.size();

      // make sure Qs are inheriting A row labels.
      outputQRow(key, qRow, aRow);

      // MAHOUT-817
      if (computeSq) {
        if (sqAccum == null) {
          sqAccum = new DenseVector(kp);
        }
        sqAccum.assign(qRow, Functions.PLUS);
      }

      if (btRow == null) {
        btRow = new DenseVector(kp);
      }

      if (!aRow.isDense()) {
        for (Vector.Element el : aRow.nonZeroes()) {
          double mul = el.get();
          for (int j = 0; j < kp; j++) {
            btRow.setQuick(j, mul * qRow.getQuick(j));
          }
          btCollector.collect((long) el.index(), btRow);
        }
      } else {
        int n = aRow.size();
        for (int i = 0; i < n; i++) {
          double mul = aRow.getQuick(i);
          for (int j = 0; j < kp; j++) {
            btRow.setQuick(j, mul * qRow.getQuick(j));
          }
          btCollector.collect((long) i, btRow);
        }
      }
    }

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {
      super.setup(context);

      Configuration conf = context.getConfiguration();

      Path qJobPath = new Path(conf.get(PROP_QJOB_PATH));

      /*
       * actually this is kind of dangerous because this routine thinks we need
       * to create file name for our current job and this will use -m- so it's
       * just serendipity we are calling it from the mapper too as the QJob did.
       */
      Path qInputPath =
        new Path(qJobPath, FileOutputFormat.getUniqueFile(context,
                                                          QJob.OUTPUT_QHAT,
                                                          ""));
      blockNum = context.getTaskAttemptID().getTaskID().getId();

      SequenceFileValueIterator<DenseBlockWritable> qhatInput =
        new SequenceFileValueIterator<DenseBlockWritable>(qInputPath,
                                                          true,
                                                          conf);
      closeables.addFirst(qhatInput);

      /*
       * read all r files _in order of task ids_, i.e. partitions (aka group
       * nums).
       *
       * Note: if broadcast option is used, this comes from distributed cache
       * files rather than hdfs path.
       */

      SequenceFileDirValueIterator<VectorWritable> rhatInput;

      boolean distributedRHat = conf.get(PROP_RHAT_BROADCAST) != null;
      if (distributedRHat) {

        Path[] rFiles = HadoopUtil.getCachedFiles(conf);

        Validate.notNull(rFiles,
                         "no RHat files in distributed cache job definition");
        //TODO: this probably can be replaced w/ local fs makeQualified
        Configuration lconf = new Configuration();
        lconf.set("fs.default.name", "file:///");

        rhatInput =
          new SequenceFileDirValueIterator<VectorWritable>(rFiles,
                                                           SSVDHelper.PARTITION_COMPARATOR,
                                                           true,
                                                           lconf);

      } else {
        Path rPath = new Path(qJobPath, QJob.OUTPUT_RHAT + "-*");
        rhatInput =
          new SequenceFileDirValueIterator<VectorWritable>(rPath,
                                                           PathType.GLOB,
                                                           null,
                                                           SSVDHelper.PARTITION_COMPARATOR,
                                                           true,
                                                           conf);
      }

      Validate.isTrue(rhatInput.hasNext(), "Empty R-hat input!");

      closeables.addFirst(rhatInput);
      outputs = new MultipleOutputs(new JobConf(conf));
      closeables.addFirst(new IOUtils.MultipleOutputsCloseableAdapter(outputs));

      qr = new QRLastStep(qhatInput, rhatInput, blockNum);
      closeables.addFirst(qr);
      /*
       * it's so happens that current QRLastStep's implementation preloads R
       * sequence into memory in the constructor so it's ok to close rhat input
       * now.
       */
      if (!rhatInput.hasNext()) {
        closeables.remove(rhatInput);
        rhatInput.close();
      }

      OutputCollector<LongWritable, SparseRowBlockWritable> btBlockCollector =
        new OutputCollector<LongWritable, SparseRowBlockWritable>() {

          @Override
          public void collect(LongWritable blockKey,
                              SparseRowBlockWritable block) throws IOException {
            try {
              mapContext.write(blockKey, block);
            } catch (InterruptedException exc) {
              throw new IOException("Interrupted.", exc);
            }
          }
        };

      btCollector =
        new SparseRowBlockAccumulator(conf.getInt(PROP_OUTER_PROD_BLOCK_HEIGHT,
                                                  -1), btBlockCollector);
      closeables.addFirst(btCollector);

      // MAHOUT-817
      computeSq = conf.get(PROP_XI_PATH) != null;

      // MAHOUT-1067
      nv = conf.getBoolean(PROP_NV, false);

    }

    @Override
    protected void cleanup(Context context) throws IOException,
      InterruptedException {
      try {
        if (sqAccum != null) {
          /*
           * hack: we will output sq partial sums with index -1 for summation.
           */
          SparseRowBlockWritable sbrw = new SparseRowBlockWritable(1);
          sbrw.plusRow(0, sqAccum);
          LongWritable lw = new LongWritable(-1);
          context.write(lw, sbrw);
        }
      } finally {
        IOUtils.close(closeables);
      }
    }

    @SuppressWarnings("unchecked")
    private void outputQRow(Writable key, Vector qRow, Vector aRow) throws IOException {
      if (nv && (aRow instanceof NamedVector)) {
        qRowValue.set(new NamedVector(qRow, ((NamedVector) aRow).getName()));
      } else {
        qRowValue.set(qRow);
      }
      outputs.getCollector(OUTPUT_Q, null).collect(key, qRowValue);
    }
  }

  public static class OuterProductCombiner
    extends
    Reducer<Writable, SparseRowBlockWritable, Writable, SparseRowBlockWritable> {

    protected final SparseRowBlockWritable accum = new SparseRowBlockWritable();
    protected final Deque<Closeable> closeables = new ArrayDeque<Closeable>();
    protected int blockHeight;

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {
      blockHeight =
        context.getConfiguration().getInt(PROP_OUTER_PROD_BLOCK_HEIGHT, -1);
    }

    @Override
    protected void reduce(Writable key,
                          Iterable<SparseRowBlockWritable> values,
                          Context context) throws IOException,
      InterruptedException {
      for (SparseRowBlockWritable bw : values) {
        accum.plusBlock(bw);
      }
      context.write(key, accum);
      accum.clear();
    }

    @Override
    protected void cleanup(Context context) throws IOException,
      InterruptedException {

      IOUtils.close(closeables);
    }
  }

  public static class OuterProductReducer
    extends
    Reducer<LongWritable, SparseRowBlockWritable, IntWritable, VectorWritable> {

    protected final SparseRowBlockWritable accum = new SparseRowBlockWritable();
    protected final Deque<Closeable> closeables = new ArrayDeque<Closeable>();

    protected int blockHeight;
    private boolean outputBBt;
    private UpperTriangular mBBt;
    private MultipleOutputs outputs;
    private final IntWritable btKey = new IntWritable();
    private final VectorWritable btValue = new VectorWritable();

    // MAHOUT-817
    private Vector xi;
    private final PlusMult pmult = new PlusMult(0);
    private Vector sbAccum;

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {

      Configuration conf = context.getConfiguration();
      blockHeight = conf.getInt(PROP_OUTER_PROD_BLOCK_HEIGHT, -1);

      outputBBt = conf.getBoolean(PROP_OUPTUT_BBT_PRODUCTS, false);

      if (outputBBt) {
        int k = conf.getInt(QJob.PROP_K, -1);
        int p = conf.getInt(QJob.PROP_P, -1);

        Validate.isTrue(k > 0, "invalid k parameter");
        Validate.isTrue(p >= 0, "invalid p parameter");
        mBBt = new UpperTriangular(k + p);

      }

      String xiPathStr = conf.get(PROP_XI_PATH);
      if (xiPathStr != null) {
        xi = SSVDHelper.loadAndSumUpVectors(new Path(xiPathStr), conf);
        if (xi == null) {
          throw new IOException(String.format("unable to load mean path xi from %s.",
                                              xiPathStr));
        }
      }

      if (outputBBt || xi != null) {
        outputs = new MultipleOutputs(new JobConf(conf));
        closeables.addFirst(new IOUtils.MultipleOutputsCloseableAdapter(outputs));
      }

    }

    @Override
    protected void reduce(LongWritable key,
                          Iterable<SparseRowBlockWritable> values,
                          Context context) throws IOException,
      InterruptedException {

      accum.clear();
      for (SparseRowBlockWritable bw : values) {
        accum.plusBlock(bw);
      }

      // MAHOUT-817:
      if (key.get() == -1L) {

        Vector sq = accum.getRows()[0];

        @SuppressWarnings("unchecked")
        OutputCollector<IntWritable, VectorWritable> sqOut =
          outputs.getCollector(OUTPUT_SQ, null);

        sqOut.collect(new IntWritable(0), new VectorWritable(sq));
        return;
      }

      /*
       * at this point, sum of rows should be in accum, so we just generate
       * outer self product of it and add to BBt accumulator.
       */

      for (int k = 0; k < accum.getNumRows(); k++) {
        Vector btRow = accum.getRows()[k];
        btKey.set((int) (key.get() * blockHeight + accum.getRowIndices()[k]));
        btValue.set(btRow);
        context.write(btKey, btValue);

        if (outputBBt) {
          int kp = mBBt.numRows();
          // accumulate partial BBt sum
          for (int i = 0; i < kp; i++) {
            double vi = btRow.get(i);
            if (vi != 0.0) {
              for (int j = i; j < kp; j++) {
                double vj = btRow.get(j);
                if (vj != 0.0) {
                  mBBt.setQuick(i, j, mBBt.getQuick(i, j) + vi * vj);
                }
              }
            }
          }
        }

        // MAHOUT-817
        if (xi != null) {
          // code defensively against shortened xi
          int btIndex = btKey.get();
          double xii = xi.size() > btIndex ? xi.getQuick(btIndex) : 0.0;
          // compute s_b
          pmult.setMultiplicator(xii);
          if (sbAccum == null) {
            sbAccum = new DenseVector(btRow.size());
          }
          sbAccum.assign(btRow, pmult);
        }

      }
    }

    @Override
    protected void cleanup(Context context) throws IOException,
      InterruptedException {

      // if we output BBt instead of Bt then we need to do it.
      try {
        if (outputBBt) {

          @SuppressWarnings("unchecked")
          OutputCollector<Writable, Writable> collector =
            outputs.getCollector(OUTPUT_BBT, null);

          collector.collect(new IntWritable(),
                            new VectorWritable(new DenseVector(mBBt.getData())));
        }

        // MAHOUT-817
        if (sbAccum != null) {
          @SuppressWarnings("unchecked")
          OutputCollector<IntWritable, VectorWritable> collector =
            outputs.getCollector(OUTPUT_SB, null);

          collector.collect(new IntWritable(), new VectorWritable(sbAccum));

        }
      } finally {
        IOUtils.close(closeables);
      }

    }
  }

  public static void run(Configuration conf,
                         Path[] inputPathA,
                         Path inputPathQJob,
                         Path xiPath,
                         Path outputPath,
                         int minSplitSize,
                         int k,
                         int p,
                         int btBlockHeight,
                         int numReduceTasks,
                         boolean broadcast,
                         Class<? extends Writable> labelClass,
                         boolean outputBBtProducts)
    throws ClassNotFoundException, InterruptedException, IOException {

    JobConf oldApiJob = new JobConf(conf);

    MultipleOutputs.addNamedOutput(oldApiJob,
                                   OUTPUT_Q,
                                   org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
                                   labelClass,
                                   VectorWritable.class);

    if (outputBBtProducts) {
      MultipleOutputs.addNamedOutput(oldApiJob,
                                     OUTPUT_BBT,
                                     org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
                                     IntWritable.class,
                                     VectorWritable.class);
      /*
       * MAHOUT-1067: if we are asked to output BBT products then named vector
       * names should be propagated to Q too so that UJob could pick them up
       * from there.
       */
      oldApiJob.setBoolean(PROP_NV, true);
    }
    if (xiPath != null) {
      // compute pca -related stuff as well
      MultipleOutputs.addNamedOutput(oldApiJob,
                                     OUTPUT_SQ,
                                     org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
                                     IntWritable.class,
                                     VectorWritable.class);
      MultipleOutputs.addNamedOutput(oldApiJob,
                                     OUTPUT_SB,
                                     org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
                                     IntWritable.class,
                                     VectorWritable.class);
    }

    /*
     * HACK: we use old api multiple outputs since they are not available in the
     * new api of either 0.20.2 or 0.20.203 but wrap it into a new api job so we
     * can use new api interfaces.
     */

    Job job = new Job(oldApiJob);
    job.setJobName("Bt-job");
    job.setJarByClass(BtJob.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.setInputPaths(job, inputPathA);
    if (minSplitSize > 0) {
      FileInputFormat.setMinInputSplitSize(job, minSplitSize);
    }
    FileOutputFormat.setOutputPath(job, outputPath);

    // WARN: tight hadoop integration here:
    job.getConfiguration().set("mapreduce.output.basename", OUTPUT_BT);

    FileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(job,
                                                      CompressionType.BLOCK);

    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(SparseRowBlockWritable.class);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(BtMapper.class);
    job.setCombinerClass(OuterProductCombiner.class);
    job.setReducerClass(OuterProductReducer.class);

    job.getConfiguration().setInt(QJob.PROP_K, k);
    job.getConfiguration().setInt(QJob.PROP_P, p);
    job.getConfiguration().set(PROP_QJOB_PATH, inputPathQJob.toString());
    job.getConfiguration().setBoolean(PROP_OUPTUT_BBT_PRODUCTS,
                                      outputBBtProducts);
    job.getConfiguration().setInt(PROP_OUTER_PROD_BLOCK_HEIGHT, btBlockHeight);

    job.setNumReduceTasks(numReduceTasks);

    /*
     * PCA-related options, MAHOUT-817
     */
    if (xiPath != null) {
      job.getConfiguration().set(PROP_XI_PATH, xiPath.toString());
    }

    /*
     * we can broadhast Rhat files since all of them are reuqired by each job,
     * but not Q files which correspond to splits of A (so each split of A will
     * require only particular Q file, each time different one).
     */

    if (broadcast) {
      job.getConfiguration().set(PROP_RHAT_BROADCAST, "y");

      FileSystem fs = FileSystem.get(inputPathQJob.toUri(), conf);
      FileStatus[] fstats =
        fs.globStatus(new Path(inputPathQJob, QJob.OUTPUT_RHAT + "-*"));
      if (fstats != null) {
        for (FileStatus fstat : fstats) {
          /*
           * new api is not enabled yet in our dependencies at this time, still
           * using deprecated one
           */
          DistributedCache.addCacheFile(fstat.getPath().toUri(),
                                        job.getConfiguration());
        }
      }
    }

    job.submit();
    job.waitForCompletion(false);

    if (!job.isSuccessful()) {
      throw new IOException("Bt job unsuccessful.");
    }
  }
}
