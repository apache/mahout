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
import java.util.regex.Matcher;

import com.google.common.collect.Lists;
import org.apache.commons.lang3.Validate;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
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
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.qr.QRFirstStep;

/**
 * Computes ABt products, then first step of QR which is pushed down to the
 * reducer.
 * 
 */
@SuppressWarnings("deprecation")
public final class ABtJob {

  public static final String PROP_BT_PATH = "ssvd.Bt.path";
  public static final String PROP_BT_BROADCAST = "ssvd.Bt.broadcast";

  private ABtJob() {
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
      Mapper<Writable, VectorWritable, SplitPartitionedWritable, SparseRowBlockWritable> {

    private SplitPartitionedWritable outKey;
    private final Deque<Closeable> closeables = new ArrayDeque<Closeable>();
    private SequenceFileDirIterator<IntWritable, VectorWritable> btInput;
    private Vector[] aCols;
    // private Vector[] yiRows;
    // private VectorWritable outValue = new VectorWritable();
    private int aRowCount;
    private int kp;
    private int blockHeight;
    private SparseRowBlockAccumulator yiCollector;

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
      } else {
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
          new SequentialAccessSparseVector(rowCount < 10000 ? 10000 : rowCount,
                                           1);
      } else if (aCols[col].size() < rowCount) {
        Vector newVec =
          new SequentialAccessSparseVector(rowCount << 1,
                                           aCols[col].getNumNondefaultElements() << 1);
        newVec.viewPart(0, aCols[col].size()).assign(aCols[col]);
        aCols[col] = newVec;
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException,
      InterruptedException {
      try {
        // yiRows= new Vector[aRowCount];

        int lastRowIndex = -1;

        while (btInput.hasNext()) {
          Pair<IntWritable, VectorWritable> btRec = btInput.next();
          int btIndex = btRec.getFirst().get();
          Vector btVec = btRec.getSecond().get();
          Vector aCol;
          if (btIndex > aCols.length || (aCol = aCols[btIndex]) == null) {
            continue;
          }
          int j = -1;
          for (Vector.Element aEl : aCol.nonZeroes()) {
            j = aEl.index();

            // outKey.setTaskItemOrdinal(j);
            // outValue.set(btVec.times(aEl.get())); // assign might work better
            // // with memory after all.
            // context.write(outKey, outValue);
            yiCollector.collect((long) j, btVec.times(aEl.get()));
          }
          if (lastRowIndex < j) {
            lastRowIndex = j;
          }
        }
        aCols = null;

        // output empty rows if we never output partial products for them
        // this happens in sparse matrices when last rows are all zeros
        // and is subsequently causing shorter Q matrix row count which we
        // probably don't want to repair there but rather here.
        Vector yDummy = new SequentialAccessSparseVector(kp);
        // outValue.set(yDummy);
        for (lastRowIndex += 1; lastRowIndex < aRowCount; lastRowIndex++) {
          // outKey.setTaskItemOrdinal(lastRowIndex);
          // context.write(outKey, outValue);

          yiCollector.collect((long) lastRowIndex, yDummy);
        }

      } finally {
        IOUtils.close(closeables);
      }
    }

    @Override
    protected void setup(final Context context) throws IOException,
      InterruptedException {

      int k =
        Integer.parseInt(context.getConfiguration().get(QRFirstStep.PROP_K));
      int p =
        Integer.parseInt(context.getConfiguration().get(QRFirstStep.PROP_P));
      kp = k + p;

      outKey = new SplitPartitionedWritable(context);
      String propBtPathStr = context.getConfiguration().get(PROP_BT_PATH);
      Validate.notNull(propBtPathStr, "Bt input is not set");
      Path btPath = new Path(propBtPathStr);

      boolean distributedBt =
        context.getConfiguration().get(PROP_BT_BROADCAST) != null;

      if (distributedBt) {

        Path[] btFiles = HadoopUtil.getCachedFiles(context.getConfiguration());

        // DEBUG: stdout
        //System.out.printf("list of files: " + btFiles);

        StringBuilder btLocalPath = new StringBuilder();
        for (Path btFile : btFiles) {
          if (btLocalPath.length() > 0) {
            btLocalPath.append(Path.SEPARATOR_CHAR);
          }
          btLocalPath.append(btFile);
        }

        btInput =
          new SequenceFileDirIterator<IntWritable, VectorWritable>(new Path(btLocalPath.toString()),
                                                                   PathType.LIST,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   context.getConfiguration());

      } else {

        btInput =
          new SequenceFileDirIterator<IntWritable, VectorWritable>(btPath,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   context.getConfiguration());
      }
      // TODO: how do i release all that stuff??
      closeables.addFirst(btInput);
      OutputCollector<LongWritable, SparseRowBlockWritable> yiBlockCollector =
        new OutputCollector<LongWritable, SparseRowBlockWritable>() {

          @Override
          public void collect(LongWritable blockKey,
                              SparseRowBlockWritable block) throws IOException {
            outKey.setTaskItemOrdinal((int) blockKey.get());
            try {
              context.write(outKey, block);
            } catch (InterruptedException exc) {
              throw new IOException("Interrupted", exc);
            }
          }
        };
      blockHeight =
        context.getConfiguration().getInt(BtJob.PROP_OUTER_PROD_BLOCK_HEIGHT,
                                          -1);
      yiCollector =
        new SparseRowBlockAccumulator(blockHeight, yiBlockCollector);
      closeables.addFirst(yiCollector);
    }

  }

  /**
   * QR first step pushed down to reducer.
   * 
   */
  public static class QRReducer
      extends
      Reducer<SplitPartitionedWritable, SparseRowBlockWritable, SplitPartitionedWritable, VectorWritable> {

    // hack: partition number formats in hadoop, copied. this may stop working
    // if it gets
    // out of sync with newer hadoop version. But unfortunately rules of forming
    // output file names are not sufficiently exposed so we need to hack it
    // if we write the same split output from either mapper or reducer.
    // alternatively, we probably can replace it by our own output file namnig
    // management
    // completely and bypass MultipleOutputs entirely.

    private static final NumberFormat NUMBER_FORMAT =
      NumberFormat.getInstance();
    static {
      NUMBER_FORMAT.setMinimumIntegerDigits(5);
      NUMBER_FORMAT.setGroupingUsed(false);
    }

    private final Deque<Closeable> closeables = Lists.newLinkedList();
    protected final SparseRowBlockWritable accum = new SparseRowBlockWritable();

    protected int blockHeight;

    protected int accumSize;
    protected int lastTaskId = -1;

    protected OutputCollector<Writable, DenseBlockWritable> qhatCollector;
    protected OutputCollector<Writable, VectorWritable> rhatCollector;
    protected QRFirstStep qr;

    @Override
    protected void setup(Context context) throws IOException,
      InterruptedException {
      blockHeight =
        context.getConfiguration().getInt(BtJob.PROP_OUTER_PROD_BLOCK_HEIGHT,
                                          -1);

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
                          Iterable<SparseRowBlockWritable> values,
                          Context context) throws IOException,
      InterruptedException {

      accum.clear();
      for (SparseRowBlockWritable bw : values) {
        accum.plusBlock(bw);
      }

      if (key.getTaskId() != lastTaskId) {
        setupBlock(context, key);
      }

      long blockBase = key.getTaskItemOrdinal() * blockHeight;
      for (int k = 0; k < accum.getNumRows(); k++) {
        Vector yiRow = accum.getRows()[k];
        key.setTaskItemOrdinal(blockBase + accum.getRowIndices()[k]);
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
     */
    private <K,V> OutputCollector<K,V> createOutputCollector(String name,
                                                             final SplitPartitionedWritable spw,
                                                             Context ctx,
                                                             Class<V> valueClass)
      throws IOException, InterruptedException {
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
    protected void cleanup(Context context) throws IOException, InterruptedException {
      IOUtils.close(closeables);
    }

  }

  public static void run(Configuration conf,
                         Path[] inputAPaths,
                         Path inputBtGlob,
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

    // MultipleOutputs
    // .addNamedOutput(oldApiJob,
    // QJob.OUTPUT_QHAT,
    // org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
    // SplitPartitionedWritable.class,
    // DenseBlockWritable.class);
    //
    // MultipleOutputs
    // .addNamedOutput(oldApiJob,
    // QJob.OUTPUT_RHAT,
    // org.apache.hadoop.mapred.SequenceFileOutputFormat.class,
    // SplitPartitionedWritable.class,
    // VectorWritable.class);

    Job job = new Job(oldApiJob);
    job.setJobName("ABt-job");
    job.setJarByClass(ABtJob.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.setInputPaths(job, inputAPaths);
    if (minSplitSize > 0) {
      FileInputFormat.setMinInputSplitSize(job, minSplitSize);
    }

    FileOutputFormat.setOutputPath(job, outputPath);

    SequenceFileOutputFormat.setOutputCompressionType(job,
                                                      CompressionType.BLOCK);

    job.setMapOutputKeyClass(SplitPartitionedWritable.class);
    job.setMapOutputValueClass(SparseRowBlockWritable.class);

    job.setOutputKeyClass(SplitPartitionedWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(ABtMapper.class);
    job.setCombinerClass(BtJob.OuterProductCombiner.class);
    job.setReducerClass(QRReducer.class);

    job.getConfiguration().setInt(QJob.PROP_AROWBLOCK_SIZE, aBlockRows);
    job.getConfiguration().setInt(BtJob.PROP_OUTER_PROD_BLOCK_HEIGHT,
                                  outerProdBlockHeight);
    job.getConfiguration().setInt(QRFirstStep.PROP_K, k);
    job.getConfiguration().setInt(QRFirstStep.PROP_P, p);
    job.getConfiguration().set(PROP_BT_PATH, inputBtGlob.toString());

    // number of reduce tasks doesn't matter. we don't actually
    // send anything to reducers.

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
          DistributedCache.addCacheFile(fstat.getPath().toUri(), conf);
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
