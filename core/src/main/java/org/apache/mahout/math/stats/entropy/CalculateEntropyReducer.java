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

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Subtracts the partial entropy.
 */
public final class CalculateEntropyReducer
    extends Reducer<NullWritable, DoubleWritable, NullWritable, DoubleWritable> {

  private static final double LOG_2 = Math.log(2.0);

  private final DoubleWritable result = new DoubleWritable();
  private long numberItems;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    numberItems = Long.parseLong(context.getConfiguration().get(Entropy.NUMBER_ITEMS_PARAM));
  }

  @Override
  protected void reduce(NullWritable key, Iterable<DoubleWritable> values, Context context)
      throws IOException, InterruptedException {
    double entropy = 0.0;
    for (DoubleWritable value : values) {
      entropy += value.get();
    }
    result.set((Math.log(numberItems) - entropy / numberItems) / LOG_2);
    context.write(key, result);
  }

}
