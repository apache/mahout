/*
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

  private final int height;
  private final OutputCollector<LongWritable, SparseRowBlockWritable> delegate;
  private long currentBlockNum = -1;
  private SparseRowBlockWritable block;
  private final LongWritable blockKeyW = new LongWritable();

  public SparseRowBlockAccumulator(int height,
                                   OutputCollector<LongWritable, SparseRowBlockWritable> delegate) {
    this.height = height;
    this.delegate = delegate;
  }

  private void flushBlock() throws IOException {
    if (block == null || block.getNumRows() == 0) {
      return;
    }
    blockKeyW.set(currentBlockNum);
    delegate.collect(blockKeyW, block);
    block.clear();
  }

  @Override
  public void collect(Long rowIndex, Vector v) throws IOException {

    long blockKey = rowIndex / height;

    if (blockKey != currentBlockNum) {
      flushBlock();
      if (block == null) {
        block = new SparseRowBlockWritable(100);
      }
      currentBlockNum = blockKey;
    }

    block.plusRow((int) (rowIndex % height), v);
  }

  @Override
  public void close() throws IOException {
    flushBlock();
  }

}
