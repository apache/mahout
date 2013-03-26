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

package org.apache.mahout.cf.taste.example.email;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class MailToRecReducer extends Reducer<Text, LongWritable, Text, NullWritable> {
  //if true, then output weight
  private boolean useCounts = true;
  /**
   * We can either ignore how many times the user interacted (boolean) or output the number of times they interacted.
   */
  public static final String USE_COUNTS_PREFERENCE = "useBooleanPreferences";

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    useCounts = context.getConfiguration().getBoolean(USE_COUNTS_PREFERENCE, true);
  }

  @Override
  protected void reduce(Text key, Iterable<LongWritable> values, Context context)
    throws IOException, InterruptedException {
    if (useCounts) {
      long sum = 0;
      for (LongWritable value : values) {
        sum++;
      }
      context.write(new Text(key.toString() + ',' + sum), null);
    } else {
      context.write(new Text(key.toString()), null);
    }
  }
}
