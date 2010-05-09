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
import org.apache.hadoop.io.VLongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

public final class ItemIDIndexReducer extends MapReduceBase implements
    Reducer<IntWritable,VLongWritable,IntWritable,VLongWritable> {
  
  @Override
  public void reduce(IntWritable index,
                     Iterator<VLongWritable> possibleItemIDs,
                     OutputCollector<IntWritable,VLongWritable> output,
                     Reporter reporter) throws IOException {
    if (possibleItemIDs.hasNext()) {
      long minimumItemID = Long.MAX_VALUE;
      while (possibleItemIDs.hasNext()) {
        long itemID = possibleItemIDs.next().get();
        if (itemID < minimumItemID) {
          minimumItemID = itemID;
        }
      }
      output.collect(index, new VLongWritable(minimumItemID));
    }
  }
  
}