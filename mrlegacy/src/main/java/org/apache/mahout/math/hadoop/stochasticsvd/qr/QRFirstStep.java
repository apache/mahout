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
package org.apache.mahout.math.hadoop.stochasticsvd.qr;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.iterator.CopyConstructorIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.DenseBlockWritable;
import org.apache.mahout.math.UpperTriangular;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

/**
 * QR first step without MR abstractions and doing it just in terms of iterators
 * and collectors. (although Collector is probably an outdated api).
 * 
 * 
 */
@SuppressWarnings("deprecation")
public class QRFirstStep implements Closeable, OutputCollector<Writable, Vector> {

  public static final String PROP_K = "ssvd.k";
  public static final String PROP_P = "ssvd.p";
  public static final String PROP_AROWBLOCK_SIZE = "ssvd.arowblock.size";

  private int kp;
  private List<double[]> yLookahead;
  private GivensThinSolver qSolver;
  private int blockCnt;
  private final DenseBlockWritable value = new DenseBlockWritable();
  private final Writable tempKey = new IntWritable();
  private MultipleOutputs outputs;
  private final Deque<Closeable> closeables = Lists.newLinkedList();
  private SequenceFile.Writer tempQw;
  private Path tempQPath;
  private final List<UpperTriangular> rSubseq = Lists.newArrayList();
  private final Configuration jobConf;

  private final OutputCollector<? super Writable, ? super DenseBlockWritable> qtHatOut;
  private final OutputCollector<? super Writable, ? super VectorWritable> rHatOut;

  public QRFirstStep(Configuration jobConf,
                     OutputCollector<? super Writable, ? super DenseBlockWritable> qtHatOut,
                     OutputCollector<? super Writable, ? super VectorWritable> rHatOut) {
    this.jobConf = jobConf;
    this.qtHatOut = qtHatOut;
    this.rHatOut = rHatOut;
    setup();
  }

  @Override
  public void close() throws IOException {
    cleanup();
  }

  public int getKP() {
    return kp;
  }

  private void flushSolver() throws IOException {
    UpperTriangular r = qSolver.getRTilde();
    double[][] qt = qSolver.getThinQtTilde();

    rSubseq.add(r);

    value.setBlock(qt);
    getTempQw().append(tempKey, value);

    /*
     * this probably should be a sparse row matrix, but compressor should get it
     * for disk and in memory we want it dense anyway, sparse random
     * implementations would be a mostly a memory management disaster consisting
     * of rehashes and GC // thrashing. (IMHO)
     */
    value.setBlock(null);
    qSolver.reset();
  }

  // second pass to run a modified version of computeQHatSequence.
  private void flushQBlocks() throws IOException {
    if (blockCnt == 1) {
      /*
       * only one block, no temp file, no second pass. should be the default
       * mode for efficiency in most cases. Sure mapper should be able to load
       * the entire split in memory -- and we don't require even that.
       */
      value.setBlock(qSolver.getThinQtTilde());
      outputQHat(value);
      outputR(new VectorWritable(new DenseVector(qSolver.getRTilde().getData(),
                                                 true)));

    } else {
      secondPass();
    }
  }

  private void outputQHat(DenseBlockWritable value) throws IOException {
    qtHatOut.collect(NullWritable.get(), value);
  }

  private void outputR(VectorWritable value) throws IOException {
    rHatOut.collect(NullWritable.get(), value);
  }

  private void secondPass() throws IOException {
    qSolver = null; // release mem
    FileSystem localFs = FileSystem.getLocal(jobConf);
    SequenceFile.Reader tempQr =
      new SequenceFile.Reader(localFs, tempQPath, jobConf);
    closeables.addFirst(tempQr);
    int qCnt = 0;
    while (tempQr.next(tempKey, value)) {
      value
        .setBlock(GivensThinSolver.computeQtHat(value.getBlock(),
                                                qCnt,
                                                new CopyConstructorIterator<UpperTriangular>(rSubseq
                                                  .iterator())));
      if (qCnt == 1) {
        /*
         * just merge r[0] <- r[1] so it doesn't have to repeat in subsequent
         * computeQHat iterators
         */
        GivensThinSolver.mergeR(rSubseq.get(0), rSubseq.remove(1));
      } else {
        qCnt++;
      }
      outputQHat(value);
    }

    assert rSubseq.size() == 1;

    outputR(new VectorWritable(new DenseVector(rSubseq.get(0).getData(), true)));

  }

  protected void map(Vector incomingYRow) throws IOException {
    double[] yRow;
    if (yLookahead.size() == kp) {
      if (qSolver.isFull()) {

        flushSolver();
        blockCnt++;

      }
      yRow = yLookahead.remove(0);

      qSolver.appendRow(yRow);
    } else {
      yRow = new double[kp];
    }

    if (incomingYRow.isDense()) {
      for (int i = 0; i < kp; i++) {
        yRow[i] = incomingYRow.get(i);
      }
    } else {
      Arrays.fill(yRow, 0);
      for (Element yEl : incomingYRow.nonZeroes()) {
        yRow[yEl.index()] = yEl.get();
      }
    }

    yLookahead.add(yRow);
  }

  protected void setup() {

    int r = Integer.parseInt(jobConf.get(PROP_AROWBLOCK_SIZE));
    int k = Integer.parseInt(jobConf.get(PROP_K));
    int p = Integer.parseInt(jobConf.get(PROP_P));
    kp = k + p;

    yLookahead = Lists.newArrayListWithCapacity(kp);
    qSolver = new GivensThinSolver(r, kp);
    outputs = new MultipleOutputs(new JobConf(jobConf));
    closeables.addFirst(new Closeable() {
      @Override
      public void close() throws IOException {
        outputs.close();
      }
    });

  }

  protected void cleanup() throws IOException {
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
        flushSolver();
        assert tempQw != null;
        closeables.remove(tempQw);
        Closeables.close(tempQw, false);
      }
      flushQBlocks();

    } finally {
      IOUtils.close(closeables);
    }

  }

  private SequenceFile.Writer getTempQw() throws IOException {
    if (tempQw == null) {
      /*
       * temporary Q output hopefully will not exceed size of IO cache in which
       * case it is only good since it is going to be managed by kernel, not
       * java GC. And if IO cache is not good enough, then at least it is always
       * sequential.
       */
      String taskTmpDir = System.getProperty("java.io.tmpdir");

      FileSystem localFs = FileSystem.getLocal(jobConf);
      Path parent = new Path(taskTmpDir);
      Path sub = new Path(parent, "qw_" + System.currentTimeMillis());
      tempQPath = new Path(sub, "q-temp.seq");
      tempQw =
        SequenceFile.createWriter(localFs,
                                  jobConf,
                                  tempQPath,
                                  IntWritable.class,
                                  DenseBlockWritable.class,
                                  CompressionType.BLOCK);
      closeables.addFirst(tempQw);
      closeables.addFirst(new IOUtils.DeleteFileOnClose(new File(tempQPath
        .toString())));
    }
    return tempQw;
  }

  @Override
  public void collect(Writable key, Vector vw) throws IOException {
    map(vw);
  }

}
