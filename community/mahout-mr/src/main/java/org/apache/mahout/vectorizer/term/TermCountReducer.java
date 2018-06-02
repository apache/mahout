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

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.vectorizer.DictionaryVectorizer;

/**
 * This accumulates all the words and the weights and sums them up.
 *
 * @see TermCountCombiner
 */
public class TermCountReducer extends Reducer<Text, LongWritable, Text, LongWritable> {

  private int minSupport;

  @Override
  protected void reduce(Text key, Iterable<LongWritable> values, Context context)
    throws IOException, InterruptedException {
    long sum = 0;
    for (LongWritable value : values) {
      sum += value.get();
    }
    if (sum >= minSupport) {
      context.write(key, new LongWritable(sum));
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    minSupport = context.getConfiguration().getInt(DictionaryVectorizer.MIN_SUPPORT,
                                                   DictionaryVectorizer.DEFAULT_MIN_SUPPORT);
  }

}
