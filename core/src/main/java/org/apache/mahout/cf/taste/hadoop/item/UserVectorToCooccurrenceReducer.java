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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class UserVectorToCooccurrenceReducer extends MapReduceBase implements
    Reducer<IndexIndexWritable,IntWritable,IntWritable,VectorWritable> {

  private int lastItem1ID = Integer.MIN_VALUE;
  private int lastItem2ID = Integer.MIN_VALUE;
  private Vector cooccurrenceRow = null;
  private int count = 0;

  @Override
  public void reduce(IndexIndexWritable entityEntity,
                     Iterator<IntWritable> counts,
                     OutputCollector<IntWritable,VectorWritable> output,
                     Reporter reporter) throws IOException {

    int item1ID = entityEntity.getAID();
    int item2ID = entityEntity.getBID();
    int sum = CooccurrenceCombiner.sum(counts);

    if (item1ID < lastItem1ID) {
      throw new IllegalStateException();
    }
    if (item1ID == lastItem1ID) {
      if (item2ID < lastItem2ID) {
        throw new IllegalStateException();
      }
      if (item2ID == lastItem2ID) {
        count += sum;
      } else {
        if (cooccurrenceRow == null) {
          cooccurrenceRow = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
        }
        cooccurrenceRow.set(lastItem2ID, count);
        lastItem2ID = item2ID;
        count = sum;
      }
    } else {
      if (cooccurrenceRow != null) {
        if (count > 0) {
          cooccurrenceRow.set(lastItem2ID, count);
        }
        VectorWritable vw = new VectorWritable(cooccurrenceRow);
        vw.setWritesLaxPrecision(true);
        output.collect(new IntWritable(lastItem1ID), vw);
      }
      lastItem1ID = item1ID;
      lastItem2ID = item2ID;
      cooccurrenceRow = null;
      count = sum;
    }
  }
  
}