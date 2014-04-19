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
import java.text.NumberFormat;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.Iterator;
import java.util.regex.Matcher;

import com.google.common.collect.Lists;
import org.apache.commons.lang3.Validate;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.stochasticsvd.qr.QRFirstStep;

/**
 * Computes ABt products, then first step of QR which is pushed down to the
 * reducer.
 */
@SuppressWarnings("deprecation")
public final class ABtDenseOutJob {

  public static final String PROP_BT_PATH = "ssvd.Bt.path";
  public static final String PROP_BT_BROADCAST = "ssvd.Bt.broadcast";
  public static final String PROP_SB_PATH = "ssvdpca.sb.path";
  public static final String PROP_SQ_PATH = "ssvdpca.sq.path";
  public static final String PROP_XI_PATH = "ssvdpca.xi.path";

  private ABtDenseOutJob() {
  }

  /**
   * So, here, i preload A block into memory.
   * <P>
   * 
   * A sparse matrix seems to be ideal for that but there are two reasons why i
   * am not using it:
   * <UL>
   * <LI>1) I don't know the full block height. so i may need to reallocate it
   * from time to time. Although this probably not a showstopper.
   * <LI>2) I found that RandomAccessSparseVectors seem to take much more memory
   * than the SequentialAccessSparseVectors.
   * </UL>
   * <P>
   * 
   */
  public static class ABtMapper
      extends
      Mapper<Writable, VectorWritable, SplitPartitionedWritable, DenseBlockWritable> {

    private SplitPartitionedWritable outKey;
    private final Deque<Closeable> closeables = new ArrayDeque<Closeable>();
    private SequenceFileDirIterator<IntWritable, VectorWritable> btInput;
    private Vector[] aCols;
    private double[][] yiCols;
    private int aRowCount;
    private int kp;
    private int blockHeight;
    private boolean distributedBt;
    private Path[] btLocalPath;
    private Configuration localFsConfig;
    /*
     * xi and s_q are PCA-related corrections, per MAHOUT-817
     */
    protected Vector xi;
    protected Vector sq;

    @Override
    protected void map(Writable key, VectorWritable value, Context context)
      throws IOException, InterruptedException {

      Vector vec = value.get();

      int vecSize = vec.size();
      if (aCols == null) {
        aCols = new Vector[vecSize];
      } else if (aCols.length < vecSize) {
        aCols = Arrays.copyOf(aCols, vecSize);
      }

      if (vec.isDense()) {
        for (int i = 0; i < vecSize; i++) {
          extendAColIfNeeded(i, aRowCount + 1);
          aCols[i].setQuick(aRowCount, vec.getQuick(i));
        }
      } else if (vec.size() > 0) {
        for (Vector.Element vecEl : vec.nonZeroes()) {
          int i = vecEl.index();
          extendAColIfNeeded(i, aRowCount + 1);
          aCols[i].setQuick(aRowCount, vecEl.get());
        }
      }
      aRowCount++;
    }

    private void extendAColIfNeeded(int col, int rowCount) {
      if (aCols[col] == null) {
        aCols[col] =
          new SequentialAccessSparseVector(rowCount < blockHeight ? blockHeight
              : rowCount, 1);
      } else if (aCols[col].size() < rowCount) {
        Vector newVec =
          new SequentialAccessSparseVector(rowCount + blockHeight,
                                           aCols[col].getNumNondefaultElements() << 1);
        newVec.viewPart(0, aCols[col].size()).assign(aCols[col]);
        aCols[col] = newVec;
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException,
      InterruptedException {
      try {

        yiCols = new double[kp][];

        for (int i = 0; i < kp; i++) {
          yiCols[i] = new double[Math.min(aRowCount, blockHeight)];
        }

        int numPasses = (aRowCount - 1) / blockHeight + 1;

        String propBtPathStr = context.getConfiguration().get(PROP_BT_PATH);
        Validate.notNull(propBtPathStr, "Bt input is not set");
        Path btPath = new Path(propBtPathStr);
        DenseBlockWritable dbw = new DenseBlockWritable();

        /*
         * so it turns out that it may be much more efficient to do a few
         * independent passes over Bt accumulating the entire block in memory
         * than pass huge amount of blocks out to combiner. so we aim of course
         * to fit entire s x (k+p) dense block in memory where s is the number
         * of A rows in this split. If A is much sparser than (k+p) avg # of
         * elements per row then the block may exceed the split size. if this
         * happens, and if the given blockHeight is not high enough to
         * accomodate this (because of memory constraints), then we start
         * splitting s into several passes. since computation is cpu-bound
         * anyway, it should be o.k. for supersparse inputs. (as ok it can be
         * that projection is thicker than the original anyway, why would one
         * use that many k+p then).
         */
        int lastRowIndex = -1;
        for (int pass = 0; pass < numPasses; pass++) {

          if (distributedBt) {

            btInput =
              new SequenceFileDirIterator<IntWritable, VectorWritable>(btLocalPath,
                                                                       true,
                                                                       localFsConfig);

          } else {

            btInput =
              new SequenceFileDirIterator<IntWritable, VectorWritable>(btPath,
                                                                       PathType.GLOB,
                                                                       null,
                                                                       null,
                                                                       true,
                                                                       context.getConfiguration());
          }
          closeables.addFirst(btInput);
          Validate.isTrue(btInput.hasNext(), "Empty B' input!");

          int aRowBegin = pass * blockHeight;
          int bh = Math.min(blockHeight, aRowCount - aRowBegin);

          /*
           * check if we need to trim block allocation
           */
          if (pass > 0) {
            if (bh == blockHeight) {
              for (int i = 0; i < kp; i++) {
                Arrays.fill(yiCols[i], 0.0);
              }
            } else {

              for (int i = 0; i < kp; i++) {
                yiCols[i] = null;
              }
              for (int i = 0; i < kp; i++) {
                yiCols[i] = new double[bh];
              }
            }
          }

          while (btInput.hasNext()) {
            Pair<IntWritable, VectorWritable> btRec = btInput.next();
            int btIndex = btRec.getFirst().get();
            Vector btVec = btRec.getSecond().get();
            Vector aCol;
            if (btIndex > aCols.length || (aCol = aCols[btIndex]) == null
                || aCol.size() == 0) {

              /* 100% zero A column in the block, skip it as sparse */
              continue;
            }
            int j = -1;
            for (Vector.Element aEl : aCol.nonZeroes()) {
              j = aEl.index();

              /*
               * now we compute only swathes between aRowBegin..aRowBegin+bh
               * exclusive. it seems like a deficiency but in fact i think it
               * will balance itself out: either A is dense and then we
               * shouldn't have more than one pass and therefore filter
               * conditions will never kick in. Or, the only situation where we
               * can't fit Y_i block in memory is when A input is much sparser
               * than k+p per row. But if this is the case, then we'd be looking
               * at very few elements without engaging them in any operations so
               * even then it should be ok.
               */
              if (j < aRowBegin) {
                continue;
              }
              if (j >= aRowBegin + bh) {
                break;
              }

              /*
               * assume btVec is dense
               */
              if (xi != null) {
                /*
                 * MAHOUT-817: PCA correction for B'. I rewrite the whole
                 * computation loop so i don't have to check if PCA correction
                 * is needed at individual element level. It looks bulkier this
                 * way but perhaps less wasteful on cpu.
                 */
                for (int s = 0; s < kp; s++) {
                  // code defensively against shortened xi
                  double xii = xi.size() > btIndex ? xi.get(btIndex) : 0.0;
                  yiCols[s][j - aRowBegin] +=
                    aEl.get() * (btVec.getQuick(s) - xii * sq.get(s));
                }
              } else {
                /*
                 * no PCA correction
                 */
                for (int s = 0; s < kp; s++) {
                  yiCols[s][j - aRowBegin] += aEl.get() * btVec.getQuick(s);
                }
              }

            }
            if (lastRowIndex < j) {
              lastRowIndex = j;
            }
          }

          /*
           * so now we have stuff in yi
           */
          dbw.setBlock(yiCols);
          outKey.setTaskItemOrdinal(pass);
          context.write(outKey, dbw);

          closeables.remove(btInput);
          btInput.close();
        }

      } finally {
        IOUtils.close(closeables);
      }
    }

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {

      Configuration conf = context.getConfiguration();
      int k = Integer.parseInt(conf.get(QRFirstStep.PROP_K));
      int p = Integer.parseInt(conf.get(QRFirstStep.PROP_P));
      kp = k + p;

      outKey = new SplitPartitionedWritable(context);

      blockHeight = conf.getInt(BtJob.PROP_OUTER_PROD_BLOCK_HEIGHT, -1);
      distributedBt = conf.get(PROP_BT_BROADCAST) != null;
      if (distributedBt) {
        btLocalPath = HadoopUtil.getCachedFiles(conf);
        localFsConfig = new Configuration();
        localFsConfig.set("fs.default.name", "file:///");
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
   * QR first step pushed down to reducer.
   * 
   */
  public static class QRReducer
    extends Reducer<SplitPartitionedWritable, DenseBlockWritable, SplitPartitionedWritable, VectorWritable> {

    /*
     * HACK: partition number formats in hadoop, copied. this may stop working
     * if it gets out of sync with newer hadoop version. But unfortunately rules
     * of forming output file names are not sufficiently exposed so we need to
     * hack it if we write the same split output from either mapper or reducer.
     * alternatively, we probably can replace it by our own output file naming
     * management completely and bypass MultipleOutputs entirely.
     */

    private static final NumberFormat NUMBER_FORMAT =
      NumberFormat.getInstance();
    static {
      NUMBER_FORMAT.setMinimumIntegerDigits(5);
      NUMBER_FORMAT.setGroupingUsed(false);
    }

    private final Deque<Closeable> closeables = Lists.newLinkedList();

    protected int blockHeight;

    protected int accumSize;
    protected int lastTaskId = -1;

    protected OutputCollector<Writable, DenseBlockWritable> qhatCollector;
    protected OutputCollector<Writable, VectorWritable> rhatCollector;
    protected QRFirstStep qr;
    protected Vector yiRow;
    protected Vector sb;

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {
      Configuration conf = context.getConfiguration();
      blockHeight = conf.getInt(BtJob.PROP_OUTER_PROD_BLOCK_HEIGHT, -1);
      String sbPathStr = conf.get(PROP_SB_PATH);

      /*
       * PCA -related corrections (MAHOUT-817)
       */
      if (sbPathStr != null) {
        sb = SSVDHelper.loadAndSumUpVectors(new Path(sbPathStr), conf);
      }
    }

    protected void setupBlock(Context context, SplitPartitionedWritable spw)
      throws InterruptedException, IOException {
      IOUtils.close(closeables);
      qhatCollector =
        createOutputCollector(QJob.OUTPUT_QHAT,
                              spw,
                              context,
                              DenseBlockWritable.class);
      rhatCollector =
        createOutputCollector(QJob.OUTPUT_RHAT,
                              spw,
                              context,
                              VectorWritable.class);
      qr =
        new QRFirstStep(context.getConfiguration(),
                        qhatCollector,
                        rhatCollector);
      closeables.addFirst(qr);
      lastTaskId = spw.getTaskId();

    }

    @Override
    protected void reduce(SplitPartitionedWritable key,
                          Iterable<DenseBlockWritable> values,
                          Context context) throws IOException,
      InterruptedException {

      if (key.getTaskId() != lastTaskId) {
        setupBlock(context, key);
      }

      Iterator<DenseBlockWritable> iter = values.iterator();
      DenseBlockWritable dbw = iter.next();
      double[][] yiCols = dbw.getBlock();
      if (iter.hasNext()) {
        throw new IOException("Unexpected extra Y_i block in reducer input.");
      }

      long blockBase = key.getTaskItemOrdinal() * blockHeight;
      int bh = yiCols[0].length;
      if (yiRow == null) {
        yiRow = new DenseVector(yiCols.length);
      }

      for (int k = 0; k < bh; k++) {
        for (int j = 0; j < yiCols.length; j++) {
          yiRow.setQuick(j, yiCols[j][k]);
        }

        key.setTaskItemOrdinal(blockBase + k);

        // pca offset correction if any
        if (sb != null) {
          yiRow.assign(sb, Functions.MINUS);
        }

        qr.collect(key, yiRow);
      }

    }

    private Path getSplitFilePath(String name,
                                  SplitPartitionedWritable spw,
                                  Context context) throws InterruptedException,
      IOException {
      String uniqueFileName = FileOutputFormat.getUniqueFile(context, name, "");
      uniqueFileName = uniqueFileName.replaceFirst("-r-", "-m-");
      uniqueFileName =
        uniqueFileName.replaceFirst("\\d+$",
                                    Matcher.quoteReplacement(NUMBER_FORMAT.format(spw.getTaskId())));
      return new Path(FileOutputFormat.getWorkOutputPath(context),
                      uniqueFileName);
    }

    /**
     * key doesn't matter here, only value does. key always gets substituted by
     * SPW.
     * 
     * @param <K>
     *          bogus
     */
    private <K, V> OutputCollector<K, V> createOutputCollector(String name,
                              final SplitPartitionedWritable spw,
                              Context ctx,
                              Class<V> valueClass) throws IOException, InterruptedException {
      Path outputPath = getSplitFilePath(name, spw, ctx);
      final SequenceFile.Writer w =
        SequenceFile.createWriter(FileSystem.get(outputPath.toUri(), ctx.getConfiguration()),
                                  ctx.getConfiguration(),
                                  outputPath,
                                  SplitPartitionedWritable.class,
                                  valueClass);
      closeables.addFirst(w);
      return new OutputCollector<K, V>() {
        @Override
        public void collect(K key, V val) throws IOException {
          w.append(spw, val);
        }
      };
    }

    @Override
    protected void cleanup(Context context) throws IOException,
      InterruptedException {

      IOUtils.close(closeables);
    }

  }

  public static void run(Configuration conf,
                         Path[] inputAPaths,
                         Path inputBtGlob,
                         Path xiPath,
                         Path sqPath,
                         Path sbPath,
                         Path outputPath,
                         int aBlockRows,
                         int minSplitSize,
                         int k,
                         int p,
                         int outerProdBlockHeight,
                         int numReduceTasks,
                         boolean broadcastBInput)
    throws ClassNotFoundException, InterruptedException, IOException {

    JobConf oldApiJob = new JobConf(conf);

    Job job = new Job(oldApiJob);
    job.setJobName("ABt-job");
    job.setJarByClass(ABtDenseOutJob.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.setInputPaths(job, inputAPaths);
    if (minSplitSize > 0) {
      FileInputFormat.setMinInputSplitSize(job, minSplitSize);
    }

    FileOutputFormat.setOutputPath(job, outputPath);

    SequenceFileOutputFormat.setOutputCompressionType(job,
                                                      CompressionType.BLOCK);

    job.setMapOutputKeyClass(SplitPartitionedWritable.class);
    job.setMapOutputValueClass(DenseBlockWritable.class);

    job.setOutputKeyClass(SplitPartitionedWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(ABtMapper.class);
    job.setReducerClass(QRReducer.class);

    job.getConfiguration().setInt(QJob.PROP_AROWBLOCK_SIZE, aBlockRows);
    job.getConfiguration().setInt(BtJob.PROP_OUTER_PROD_BLOCK_HEIGHT,
                                  outerProdBlockHeight);
    job.getConfiguration().setInt(QRFirstStep.PROP_K, k);
    job.getConfiguration().setInt(QRFirstStep.PROP_P, p);
    job.getConfiguration().set(PROP_BT_PATH, inputBtGlob.toString());

    /*
     * PCA-related options, MAHOUT-817
     */
    if (xiPath != null) {
      job.getConfiguration().set(PROP_XI_PATH, xiPath.toString());
      job.getConfiguration().set(PROP_SB_PATH, sbPath.toString());
      job.getConfiguration().set(PROP_SQ_PATH, sqPath.toString());
    }

    job.setNumReduceTasks(numReduceTasks);

    // broadcast Bt files if required.
    if (broadcastBInput) {
      job.getConfiguration().set(PROP_BT_BROADCAST, "y");

      FileSystem fs = FileSystem.get(inputBtGlob.toUri(), conf);
      FileStatus[] fstats = fs.globStatus(inputBtGlob);
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
      throw new IOException("ABt job unsuccessful.");
    }

  }

}
