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

package org.apache.mahout.math.stats.entropy;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;

/**
 * The analog of {@link org.apache.hadoop.mapreduce.lib.reduce.IntSumReducer} which uses {@link VarIntWritable}.
 */
public final class VarIntSumReducer extends Reducer<Writable, VarIntWritable, Writable, VarIntWritable> {

  private final VarIntWritable result = new VarIntWritable();

  @Override
  protected void reduce(Writable key, Iterable<VarIntWritable> values, Context context)
      throws IOException, InterruptedException {
    int sum = 0;
    for (VarIntWritable value : values) {
      sum += value.get();
    }
    result.set(sum);
    context.write(key, result);
  }

}
