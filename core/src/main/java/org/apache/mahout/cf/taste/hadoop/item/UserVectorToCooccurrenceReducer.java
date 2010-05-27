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

import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class UserVectorToCooccurrenceReducer extends
    Reducer<VarIntWritable,VarIntWritable,VarIntWritable,VectorWritable> {

  @Override
  protected void reduce(VarIntWritable itemIndex1,
                        Iterable<VarIntWritable> itemIndex2s,
                        Context context) throws IOException, InterruptedException {
    Vector cooccurrenceRow = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    for (VarIntWritable varIntWritable : itemIndex2s) {
      int itemIndex2 = varIntWritable.get();
      cooccurrenceRow.set(itemIndex2, cooccurrenceRow.get(itemIndex2) + 1.0);
    }
    VectorWritable vw = new VectorWritable(cooccurrenceRow);
    vw.setWritesLaxPrecision(true);
    context.write(itemIndex1, vw);
  }
  
}