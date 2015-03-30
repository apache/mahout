package org.apache.mahout.math.hadoop.stats;
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

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class StandardDeviationCalculatorMapper extends
        Mapper<IntWritable, Writable, IntWritable, DoubleWritable> {

  public static final IntWritable SUM_OF_SQUARES = new IntWritable(1);
  public static final IntWritable SUM = new IntWritable(2);
  public static final IntWritable TOTAL_COUNT = new IntWritable(-1);

  @Override
  protected void map(IntWritable key, Writable value, Context context)
    throws IOException, InterruptedException {
    if (key.get() == -1) {
      return;
    }
    //Kind of ugly, but such is life
    double df = Double.NaN;
    if (value instanceof LongWritable) {
      df = ((LongWritable)value).get();
    } else if (value instanceof DoubleWritable) {
      df = ((DoubleWritable)value).get();
    }
    if (!Double.isNaN(df)) {
      // For calculating the sum of squares
      context.write(SUM_OF_SQUARES, new DoubleWritable(df * df));
      context.write(SUM, new DoubleWritable(df));
      // For calculating the total number of entries
      context.write(TOTAL_COUNT, new DoubleWritable(1));
    }
  }
}
