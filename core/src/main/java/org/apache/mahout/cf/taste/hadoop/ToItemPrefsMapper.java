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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import java.io.IOException;
import java.util.regex.Pattern;

/**
 * <h1>Input</h1>
 *
 * <p>Intended for use with {@link org.apache.hadoop.mapred.TextInputFormat}; accepts
 * line number / line pairs as {@link LongWritable}/{@link Text} pairs.</p>
 *
 * <p>Each line is assumed to be of the form <code>userID,itemID,preference</code>.</p>
 *
 * <h1>Output</h1>
 *
 * <p>Outputs the user ID as a {@link LongWritable} mapped to the item ID and preference
 * as a {@link ItemPrefWritable}.</p>
 */
public final class ToItemPrefsMapper
    extends MapReduceBase
    implements Mapper<LongWritable, Text, LongWritable, ItemPrefWritable> {

  private static final Pattern COMMA = Pattern.compile(",");

  @Override
  public void map(LongWritable key,
                  Text value,
                  OutputCollector<LongWritable, ItemPrefWritable> output,
                  Reporter reporter) throws IOException {
    String[] tokens = COMMA.split(value.toString());
    long userID = Long.parseLong(tokens[0]);
    long itemID = Long.parseLong(tokens[1]);
    float prefValue = Float.parseFloat(tokens[2]);
    output.collect(new LongWritable(userID), new ItemPrefWritable(itemID, prefValue));
  }

}