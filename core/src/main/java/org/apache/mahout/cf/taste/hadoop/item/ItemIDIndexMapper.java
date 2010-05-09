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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.VLongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

public final class ItemIDIndexMapper extends MapReduceBase implements
    Mapper<LongWritable,Text,IntWritable,VLongWritable> {
  
  private static final Pattern COMMA = Pattern.compile(",");
  
  @Override
  public void map(LongWritable key,
                  Text value,
                  OutputCollector<IntWritable,VLongWritable> output,
                  Reporter reporter) throws IOException {
    String[] tokens = ItemIDIndexMapper.COMMA.split(value.toString());
    long itemID = Long.parseLong(tokens[1]);
    int index = idToIndex(itemID);
    output.collect(new IntWritable(index), new VLongWritable(itemID));
  }
  
  static int idToIndex(long itemID) {
    return 0x7FFFFFFF & ((int) itemID ^ (int) (itemID >>> 32));
  }
  
}