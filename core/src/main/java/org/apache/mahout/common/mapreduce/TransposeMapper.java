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

package org.apache.mahout.common.mapreduce;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

public class TransposeMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

  @Override
  protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
    int row = r.get();
    for (Vector.Element e : v.get().nonZeroes()) {
      RandomAccessSparseVector tmp = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
      tmp.setQuick(row, e.get());
      r.set(e.index());
      ctx.write(r, new VectorWritable(tmp));
    }
  }
}
