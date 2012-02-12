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

package org.apache.mahout.clustering;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

public class CIReducer extends Reducer<IntWritable,ClusterWritable,IntWritable,ClusterWritable> {
  
  @Override
  protected void reduce(IntWritable key, Iterable<ClusterWritable> values, Context context) throws IOException,
      InterruptedException {
    Iterator<ClusterWritable> iter = values.iterator();
    ClusterWritable first = null;
    while (iter.hasNext()) {
      ClusterWritable cw = iter.next();
      if (first == null) {
        first = cw;
      } else {
        first.getValue().observe(cw.getValue());
      }
    }
    first.getValue().computeParameters();
    context.write(key, first);
  }
  
}
