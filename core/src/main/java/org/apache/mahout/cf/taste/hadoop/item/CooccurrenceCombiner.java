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

public final class CooccurrenceCombiner extends MapReduceBase implements
    Reducer<IndexIndexWritable,IntWritable,IndexIndexWritable,IntWritable> {

  private IndexIndexWritable lastEntityEntity =
      new IndexIndexWritable(Integer.MIN_VALUE, Integer.MIN_VALUE);
  private int count = 0;

  @Override
  public void reduce(IndexIndexWritable entityEntity,
                     Iterator<IntWritable> counts,
                     OutputCollector<IndexIndexWritable,IntWritable> output,
                     Reporter reporter) throws IOException {
    if (entityEntity.equals(lastEntityEntity)) {
      count += sum(counts);
    } else {
      if (count > 0) {
        output.collect(lastEntityEntity, new IntWritable(count));
      }
      count = sum(counts);     
      lastEntityEntity = entityEntity.clone();
    }
  }

  static int sum(Iterator<IntWritable> it) {
    int sum = 0;
    while (it.hasNext()) {
      sum += it.next().get();
    }
    return sum;
  }

}