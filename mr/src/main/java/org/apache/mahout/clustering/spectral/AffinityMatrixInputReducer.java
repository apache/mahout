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

package org.apache.mahout.clustering.spectral;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tasked with taking each DistributedRowMatrix entry and collecting them
 * into vectors corresponding to rows. The input and output keys are the same,
 * corresponding to the row in the ensuing matrix. The matrix entries are
 * entered into a vector according to the column to which they belong, and
 * the vector is then given the key corresponding to its row.
 */
public class AffinityMatrixInputReducer
    extends Reducer<IntWritable, DistributedRowMatrix.MatrixEntryWritable, IntWritable, VectorWritable> {

  private static final Logger log = LoggerFactory.getLogger(AffinityMatrixInputReducer.class);

  @Override
  protected void reduce(IntWritable row, Iterable<DistributedRowMatrix.MatrixEntryWritable> values, Context context)
    throws IOException, InterruptedException {
    int size = context.getConfiguration().getInt(Keys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE);
    RandomAccessSparseVector out = new RandomAccessSparseVector(size, 100);

    for (DistributedRowMatrix.MatrixEntryWritable element : values) {
      out.setQuick(element.getCol(), element.getVal());
      if (log.isDebugEnabled()) {
        log.debug("(DEBUG - REDUCE) Row[{}], Column[{}], Value[{}]",
                  row.get(), element.getCol(), element.getVal());
      }
    }
    SequentialAccessSparseVector output = new SequentialAccessSparseVector(out);
    context.write(row, new VectorWritable(output));
  }
}
