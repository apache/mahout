package org.apache.mahout.math.hadoop.stochasticsvd;

import java.io.Closeable;
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.math.Vector;

/**
 * Aggregate incoming rows into blocks based on the row number (long). Rows can
 * be sparse (meaning they come perhaps in big intervals) and don't even have to
 * come in any order, but they should be coming in proximity, so when we output
 * block key, we hopefully aggregate more than one row by then.
 * <P>
 * 
 * If block is sufficiently large to fit all rows that mapper may produce, it
 * will not even ever hit a spill at all as we would already be plussing
 * efficiently in the mapper.
 * <P>
 * 
 * Also, for sparse inputs it will also be working especially well if transposed
 * columns of the left side matrix and corresponding rows of the right side
 * matrix experience sparsity in same elements.
 * <P>
 * 
 */
public class SparseRowBlockAccumulator implements
    OutputCollector<Long, Vector>, Closeable {

  private int height;
  private OutputCollector<LongWritable, SparseRowBlockWritable> delegate;
  private long currentBlockNum = -1;
  private SparseRowBlockWritable block;
  private LongWritable blockKeyW = new LongWritable();

  public SparseRowBlockAccumulator(int height,
                                   OutputCollector<LongWritable, SparseRowBlockWritable> delegate) {
    super();
    this.height = height;
    this.delegate = delegate;
  }

  private void flushBlock() throws IOException {
    if (block == null || block.getNumRows() == 0)
      return;
    blockKeyW.set(currentBlockNum);
    delegate.collect(blockKeyW, block);
    block.clear();
  }

  @Override
  public void collect(Long rowIndex, Vector v) throws IOException {

    long blockKey = rowIndex / height;

    if (blockKey != currentBlockNum) {
      flushBlock();
      if (block == null)
        block = new SparseRowBlockWritable(100);
      currentBlockNum = blockKey;
    }

    block.plusRow((int) (rowIndex % height), v);
  }

  @Override
  public void close() throws IOException {
    flushBlock();
  }

}
