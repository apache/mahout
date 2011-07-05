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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;

/**
 * Does the weighted conditional entropy calculation with
 * <p/>
 * H(values|key) = p(key) * sum_i(p(values_i|key) * log_2(p(values_i|key)))
 * = p(key) * (log(|key|) - sum_i(values_i * log_2(values_i)) / |key|)
 * = (sum * log_2(sum) - sum_i(values_i * log_2(values_i))/n WITH sum = sum_i(values_i)
 * = (sum * log(sum) - sum_i(values_i * log(values_i)) / (n * log(2))
 */
public final class SpecificConditionalEntropyReducer extends Reducer<Text, VarIntWritable, Text, DoubleWritable> {

  private final DoubleWritable result = new DoubleWritable();
  private double numberItemsLog2;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    numberItemsLog2 =
        Math.log(2) * Integer.parseInt(context.getConfiguration().get(ConditionalEntropy.NUMBER_ITEMS_PARAM));
  }

  @Override
  protected void reduce(Text key, Iterable<VarIntWritable> values, Context context)
      throws IOException, InterruptedException {
    double sum = 0.0;
    double entropy = 0.0;
    for (VarIntWritable value : values) {
      int valueInt = value.get();
      sum += valueInt;
      entropy += valueInt * Math.log(valueInt);
    }
    result.set((sum * Math.log(sum) - entropy) / numberItemsLog2);
    context.write(key, result);
  }

}
