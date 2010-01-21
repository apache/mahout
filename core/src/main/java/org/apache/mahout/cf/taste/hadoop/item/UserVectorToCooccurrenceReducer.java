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

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Iterator;

public final class UserVectorToCooccurrenceReducer
    extends MapReduceBase
    implements Reducer<IntWritable, IntWritable, IntWritable, VectorWritable> {

  private final VectorWritable vectorWritable = new VectorWritable();

  @Override
  public void reduce(IntWritable index1,
                     Iterator<IntWritable> index2s,
                     OutputCollector<IntWritable, VectorWritable> output,
                     Reporter reporter) throws IOException {
    if (index2s.hasNext()) {
      RandomAccessSparseVector cooccurrenceRow = new RandomAccessSparseVector(Integer.MAX_VALUE, 1000);
      while (index2s.hasNext()) {
        int index2 = index2s.next().get();
        cooccurrenceRow.set(index2, cooccurrenceRow.get(index2) + 1.0);
      }
      Iterator<Vector.Element> cooccurrences = cooccurrenceRow.iterateNonZero();
      while (cooccurrences.hasNext()) {
        Vector.Element element = cooccurrences.next();
        if (element.get() <= 1.0) { // purge small values
          element.set(0.0);
        }
      }
      vectorWritable.set(cooccurrenceRow);
      output.collect(index1, vectorWritable);
    }
  }

}