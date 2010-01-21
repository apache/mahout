/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import EDU.oswego.cs.dl.util.concurrent.FJTask;
import EDU.oswego.cs.dl.util.concurrent.FJTaskRunnerGroup;

import org.apache.mahout.math.matrix.DoubleMatrix2D;

class Smp {

  protected final FJTaskRunnerGroup taskGroup; // a very efficient and light weight thread pool

  private final int maxThreads;

  /** Constructs a new Smp using a maximum of <tt>maxThreads<tt> threads. */
  protected Smp(int maxThreads) {
    maxThreads = Math.max(1, maxThreads);
    this.maxThreads = maxThreads;
    if (maxThreads > 1) {
      this.taskGroup = new FJTaskRunnerGroup(maxThreads);
    } else { // avoid parallel overhead
      this.taskGroup = null;
    }
  }

  /** Clean up deamon threads, if necessary. */
  @Override
  protected void finalize() throws Throwable {
    if (this.taskGroup != null) {
      this.taskGroup.interruptAll();
    }
    super.finalize();
  }

  protected void run(final DoubleMatrix2D[] blocksA, final DoubleMatrix2D[] blocksB, final double[] results,
                     final Matrix2DMatrix2DFunction function) {
    final FJTask[] subTasks = new FJTask[blocksA.length];
    for (int i = 0; i < blocksA.length; i++) {
      final int k = i;
      subTasks[i] = new FJTask() {
        @Override
        public void run() {
          double result = function.apply(blocksA[k], blocksB != null ? blocksB[k] : null);
          if (results != null) {
            results[k] = result;
          }
          //log.info(".");
        }
      };
    }

    // run tasks and wait for completion
    try {
      this.taskGroup.invoke(
          new FJTask() {
            @Override
            public void run() {
              coInvoke(subTasks);
            }
          }
      );
    } catch (InterruptedException exc) {
    }
  }

  protected DoubleMatrix2D[] splitBlockedNN(DoubleMatrix2D A, int threshold, long flops) {
    /*
    determine how to split and parallelize best into blocks
    if more B.columns than tasks --> split B.columns, as follows:

        xx|xx|xxx B
        xx|xx|xxx
        xx|xx|xxx
    A
    xxx     xx|xx|xxx C
    xxx    xx|xx|xxx
    xxx    xx|xx|xxx
    xxx    xx|xx|xxx
    xxx    xx|xx|xxx

    if less B.columns than tasks --> split A.rows, as follows:

        xxxxxxx B
        xxxxxxx
        xxxxxxx
    A
    xxx     xxxxxxx C
    xxx     xxxxxxx
    ---     -------
    xxx     xxxxxxx
    xxx     xxxxxxx
    ---     -------
    xxx     xxxxxxx

    */
    //long flops = 2L*A.rows()*A.columns()*A.columns();
    int noOfTasks =
        (int) Math.min(flops / threshold, this.maxThreads); // each thread should process at least 30000 flops
    boolean splitHoriz = (A.columns() < noOfTasks);
    //boolean splitHoriz = (A.columns() >= noOfTasks);
    int p = splitHoriz ? A.rows() : A.columns();
    noOfTasks = Math.min(p, noOfTasks);

    if (noOfTasks < 2) { // parallelization doesn't pay off (too much start up overhead)
      return null;
    }

    // set up concurrent tasks
    int span = p / noOfTasks;
    DoubleMatrix2D[] blocks = new DoubleMatrix2D[noOfTasks];
    for (int i = 0; i < noOfTasks; i++) {
      int offset = i * span;
      if (i == noOfTasks - 1) {
        span = p - span * i;
      } // last span may be a bit larger

      //DoubleMatrix2D AA, BB, CC;
      if (!splitHoriz) {   // split B along columns into blocks
        blocks[i] = A.viewPart(0, offset, A.rows(), span);
      } else { // split A along rows into blocks
        blocks[i] = A.viewPart(offset, 0, span, A.columns());
      }
    }
    return blocks;
  }

  protected DoubleMatrix2D[][] splitBlockedNN(DoubleMatrix2D A, DoubleMatrix2D B, int threshold, long flops) {
    DoubleMatrix2D[] blocksA = splitBlockedNN(A, threshold, flops);
    if (blocksA == null) {
      return null;
    }
    DoubleMatrix2D[] blocksB = splitBlockedNN(B, threshold, flops);
    if (blocksB == null) {
      return null;
    }
    return new DoubleMatrix2D[][]{blocksA, blocksB};
  }

  protected DoubleMatrix2D[] splitStridedNN(DoubleMatrix2D A, int threshold, long flops) {
    /*
    determine how to split and parallelize best into blocks
    if more B.columns than tasks --> split B.columns, as follows:

        xx|xx|xxx B
        xx|xx|xxx
        xx|xx|xxx
    A
    xxx     xx|xx|xxx C
    xxx    xx|xx|xxx
    xxx    xx|xx|xxx
    xxx    xx|xx|xxx
    xxx    xx|xx|xxx

    if less B.columns than tasks --> split A.rows, as follows:

        xxxxxxx B
        xxxxxxx
        xxxxxxx
    A
    xxx     xxxxxxx C
    xxx     xxxxxxx
    ---     -------
    xxx     xxxxxxx
    xxx     xxxxxxx
    ---     -------
    xxx     xxxxxxx

    */
    //long flops = 2L*A.rows()*A.columns()*A.columns();
    int noOfTasks =
        (int) Math.min(flops / threshold, this.maxThreads); // each thread should process at least 30000 flops
    boolean splitHoriz = (A.columns() < noOfTasks);
    //boolean splitHoriz = (A.columns() >= noOfTasks);
    int p = splitHoriz ? A.rows() : A.columns();
    noOfTasks = Math.min(p, noOfTasks);

    if (noOfTasks < 2) { // parallelization doesn't pay off (too much start up overhead)
      return null;
    }

    // set up concurrent tasks
    int span = p / noOfTasks;
    DoubleMatrix2D[] blocks = new DoubleMatrix2D[noOfTasks];
    for (int i = 0; i < noOfTasks; i++) {
      //int offset = i * span;
      if (i == noOfTasks - 1) {
        span = p - span * i;
      } // last span may be a bit larger

      //DoubleMatrix2D AA, BB, CC;
      if (!splitHoriz) {
        // split B along columns into blocks
        blocks[i] = A.viewPart(0, i, A.rows(), A.columns() - i).viewStrides(1, noOfTasks);
      } else {
        // split A along rows into blocks
        blocks[i] = A.viewPart(i, 0, A.rows() - i, A.columns()).viewStrides(noOfTasks, 1);
      }
    }
    return blocks;
  }

  /** Prints various snapshot statistics to System.out; Simply delegates to {@link EDU.oswego.cs.dl.util.concurrent.FJTaskRunnerGroup#stats}. */
  public void stats() {
    if (this.taskGroup != null) {
      this.taskGroup.stats();
    }
  }
}
