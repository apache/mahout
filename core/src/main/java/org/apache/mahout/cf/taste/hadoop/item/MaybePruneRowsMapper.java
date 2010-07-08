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
import java.util.Collections;
import java.util.Iterator;
import java.util.PriorityQueue;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntIntHashMap;

public class MaybePruneRowsMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private int maxSimilaritiesPerItemConsidered;
    private final OpenIntIntHashMap indexCounts = new OpenIntIntHashMap();

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      super.setup(ctx);
      maxSimilaritiesPerItemConsidered =
          ctx.getConfiguration().getInt(RecommenderJob.MAX_SIMILARITIES_PER_ITEM_CONSIDERED, -1);
      if (maxSimilaritiesPerItemConsidered < 1) {
        throw new IllegalStateException("Maximum number of similarities per item was not correctly set!");
      }
    }

    @Override
    protected void map(IntWritable rowIndex, VectorWritable vectorWritable, Context ctx)
        throws IOException, InterruptedException {
      Vector vector = vectorWritable.get();
      countSeen(vector);
      vector = maybePruneVector(vector);
      vectorWritable.set(vector);
      vectorWritable.setWritesLaxPrecision(true);
      ctx.write(rowIndex, vectorWritable);
    }

    private void countSeen(Vector vector) {
      Iterator<Vector.Element> it = vector.iterateNonZero();
      while (it.hasNext()) {
        int index = it.next().index();
        indexCounts.adjustOrPutValue(index, 1, 1);
      }
    }

    private Vector maybePruneVector(Vector vector) {
      if (vector.getNumNondefaultElements() <= maxSimilaritiesPerItemConsidered) {
        return vector;
      }

      PriorityQueue<Integer> smallCounts =
          new PriorityQueue<Integer>(maxSimilaritiesPerItemConsidered + 1, Collections.reverseOrder());
      Iterator<Vector.Element> it = vector.iterateNonZero();
      while (it.hasNext()) {
        int count = indexCounts.get(it.next().index());
        if (count > 0) {
          if (smallCounts.size() < maxSimilaritiesPerItemConsidered) {
            smallCounts.add(count);
          } else if (count < smallCounts.peek()) {
            smallCounts.add(count);
            smallCounts.poll();
          }
       }
     }

     int greatestSmallCount = smallCounts.peek();
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
