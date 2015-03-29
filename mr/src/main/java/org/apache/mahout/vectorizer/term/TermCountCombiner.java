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

package org.apache.mahout.vectorizer.term;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * @see TermCountReducer
 */
public class TermCountCombiner extends Reducer<Text, LongWritable, Text, LongWritable> {

  @Override
  protected void reduce(Text key, Iterable<LongWritable> values, Context context)
    throws IOException, InterruptedException {
    long sum = 0;
    for (LongWritable value : values) {
      sum += value.get();
    }
    context.write(key, new LongWritable(sum));
  }

}
