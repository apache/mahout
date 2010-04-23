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
import org.apache.mahout.cf.taste.hadoop.EntityCountWritable;
import org.apache.mahout.math.function.IntIntProcedure;
import org.apache.mahout.math.map.OpenIntIntHashMap;

public final class CooccurrenceCombiner extends MapReduceBase implements
    Reducer<IntWritable,EntityCountWritable,IntWritable,EntityCountWritable> {

  @Override
  public void reduce(final IntWritable index1,
                     Iterator<EntityCountWritable> index2s,
                     final OutputCollector<IntWritable,EntityCountWritable> output,
                     Reporter reporter) {

    OpenIntIntHashMap indexCounts = new OpenIntIntHashMap();
    while (index2s.hasNext()) {
      EntityCountWritable writable = index2s.next();
      int index = (int) writable.getID();
      int oldValue = indexCounts.get(index);
      indexCounts.put(index, oldValue + writable.getCount());
    }

    final EntityCountWritable entityCount = new EntityCountWritable();
    indexCounts.forEachPair(new IntIntProcedure() {
      @Override
      public boolean apply(int index, int count) {
        entityCount.set(index, count);
        try {
          output.collect(index1, entityCount);
        } catch (IOException ioe) {
          throw new IllegalStateException(ioe);
        }
        return true;
      }
    });
  }
}