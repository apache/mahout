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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.common.MinK;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.map.OpenIntIntHashMap;

import java.io.IOException;
import java.util.Comparator;
import java.util.Iterator;


/**
 * tries to limit the number of elements per col to a fixed size and transposes the input afterwards
 */
public class MaybePruneRowsMapper
    extends Mapper<VarLongWritable,VectorWritable,IntWritable,DistributedRowMatrix.MatrixEntryWritable> {

  public static final String MAX_COOCCURRENCES = MaybePruneRowsMapper.class.getName() + ".maxCooccurrences";
  
  private int maxCooccurrences;
  private final OpenIntIntHashMap indexCounts = new OpenIntIntHashMap();

  enum Elements {
    USED, NEGLECTED
  }

  @Override
  protected void setup(Context ctx) throws IOException, InterruptedException {
    super.setup(ctx);
    maxCooccurrences = ctx.getConfiguration().getInt(MAX_COOCCURRENCES, -1);
    if (maxCooccurrences < 1) {
      throw new IllegalStateException("Maximum number of cooccurrences was not correctly set!");
    }
  }

  @Override
  protected void map(VarLongWritable rowIndex, VectorWritable vectorWritable, Context ctx)
    throws IOException, InterruptedException {
    Vector vector = vectorWritable.get();
    countSeen(vector);

    int numElementsBeforePruning = vector.getNumNondefaultElements();
    vector = maybePruneVector(vector);
    int numElementsAfterPruning = vector.getNumNondefaultElements();

    ctx.getCounter(Elements.USED).increment(numElementsAfterPruning);
    ctx.getCounter(Elements.NEGLECTED).increment(numElementsBeforePruning - numElementsAfterPruning);

    DistributedRowMatrix.MatrixEntryWritable entry = new DistributedRowMatrix.MatrixEntryWritable();
    int colIndex = TasteHadoopUtils.idToIndex(rowIndex.get());
    entry.setCol(colIndex);
    Iterator<Vector.Element> iterator = vector.iterateNonZero();
    while (iterator.hasNext()) {
      Vector.Element elem = iterator.next();
      entry.setRow(elem.index());
      entry.setVal(elem.get());
      ctx.write(new IntWritable(elem.index()), entry);
    }
  }

  private void countSeen(Vector vector) {
    Iterator<Vector.Element> it = vector.iterateNonZero();
    while (it.hasNext()) {
      int index = it.next().index();
      indexCounts.adjustOrPutValue(index, 1, 1);
    }
  }

  private Vector maybePruneVector(Vector vector) {
    if (vector.getNumNondefaultElements() <= maxCooccurrences) {
      return vector;
    }

    MinK<Integer> smallCounts = new MinK<Integer>(maxCooccurrences, new Comparator<Integer>() {
        @Override
        public int compare(Integer one, Integer two) {
          return one.compareTo(two);
        }
      });

    Iterator<Vector.Element> it = vector.iterateNonZero();
    while (it.hasNext()) {
      int count = indexCounts.get(it.next().index());
      smallCounts.offer(count);
    }

    int greatestSmallCount = smallCounts.greatestSmall();
    if (greatestSmallCount > 0) {
      Iterator<Vector.Element> it2 = vector.iterateNonZero();
      while (it2.hasNext()) {
        Vector.Element e = it2.next();
        if (indexCounts.get(e.index()) > greatestSmallCount) {
          e.set(0.0);
        }
      }
    }
    return vector;
  }
}
