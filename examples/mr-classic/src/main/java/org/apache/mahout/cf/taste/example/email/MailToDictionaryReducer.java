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

package org.apache.mahout.cf.taste.example.email;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;

/**
 * Key: the string id
 * Value: the count
 * Out Key: the string id
 * Out Value: the sum of the counts
 */
public final class MailToDictionaryReducer extends Reducer<Text, VarIntWritable, Text, VarIntWritable> {

  @Override
  protected void reduce(Text key, Iterable<VarIntWritable> values, Context context)
    throws IOException, InterruptedException {
    int sum = 0;
    for (VarIntWritable value : values) {
      sum += value.get();
    }
    context.write(new Text(key), new VarIntWritable(sum));
  }
}
