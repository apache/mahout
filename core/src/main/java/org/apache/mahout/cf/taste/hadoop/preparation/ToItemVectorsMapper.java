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

package org.apache.mahout.cf.taste.hadoop.preparation;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

public class ToItemVectorsMapper
    extends Mapper<VarLongWritable,VectorWritable,IntWritable,VectorWritable> {

  private final IntWritable itemID = new IntWritable();
  private final VectorWritable itemVectorWritable = new VectorWritable();

  @Override
  protected void map(VarLongWritable rowIndex, VectorWritable vectorWritable, Context ctx)
    throws IOException, InterruptedException {
    Vector userRatings = vectorWritable.get();

    int column = TasteHadoopUtils.idToIndex(rowIndex.get());

    itemVectorWritable.setWritesLaxPrecision(true);

    Vector itemVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
    for (Vector.Element elem : userRatings.nonZeroes()) {
      itemID.set(elem.index());
      itemVector.setQuick(column, elem.get());
      itemVectorWritable.set(itemVector);
      ctx.write(itemID, itemVectorWritable);
      // reset vector for reuse
      itemVector.setQuick(elem.index(), 0.0);
    }
  }

}
