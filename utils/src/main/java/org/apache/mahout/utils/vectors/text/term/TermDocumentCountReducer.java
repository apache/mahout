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

package org.apache.mahout.utils.vectors.text.term;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

/**
 * Can also be used as a local Combiner. This accumulates all the features and the weights and sums them up.
 */
public class TermDocumentCountReducer extends MapReduceBase implements
    Reducer<IntWritable,LongWritable,IntWritable,LongWritable> {
  
  @Override
  public void reduce(IntWritable key,
                     Iterator<LongWritable> values,
                     OutputCollector<IntWritable,LongWritable> output,
                     Reporter reporter) throws IOException {
    long sum = 0;
    while (values.hasNext()) {
      sum += values.next().get();
    }
    output.collect(key, new LongWritable(sum));
  }
}
