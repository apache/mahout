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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class UserVectorToCooccurrenceReducer extends MapReduceBase implements
    Reducer<VarIntWritable,VarIntWritable,VarIntWritable,VectorWritable> {

  @Override
  public void reduce(VarIntWritable itemIndex1,
                     Iterator<VarIntWritable> itemIndex2s,
                     OutputCollector<VarIntWritable,VectorWritable> output,
                     Reporter reporter) throws IOException {
    Vector cooccurrenceRow = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    while (itemIndex2s.hasNext()) {
      int itemIndex2 = itemIndex2s.next().get();
      cooccurrenceRow.set(itemIndex2, cooccurrenceRow.get(itemIndex2) + 1.0);
    }
    VectorWritable vw = new VectorWritable(cooccurrenceRow);
    vw.setWritesLaxPrecision(true);
    output.collect(itemIndex1, vw);
  }
  
}