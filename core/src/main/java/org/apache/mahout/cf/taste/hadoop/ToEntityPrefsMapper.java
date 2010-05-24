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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.item.RecommenderJob;
import org.apache.mahout.math.VarLongWritable;

import java.io.IOException;
import java.util.regex.Pattern;

public abstract class ToEntityPrefsMapper extends MapReduceBase implements
    Mapper<LongWritable,Text, VarLongWritable,VarLongWritable> {

  public static final String TRANSPOSE_USER_ITEM = "transposeUserItem";

  private static final Pattern DELIMITER = Pattern.compile("[\t,]");

  private boolean booleanData;
  private boolean transpose;
  private final boolean itemKey;

  ToEntityPrefsMapper(boolean itemKey) {
    this.itemKey = itemKey;
  }

  @Override
  public void configure(JobConf jobConf) {
    booleanData = jobConf.getBoolean(RecommenderJob.BOOLEAN_DATA, false);
    transpose = jobConf.getBoolean(TRANSPOSE_USER_ITEM, false);
  }

  @Override
  public void map(LongWritable key,
                  Text value,
                  OutputCollector<VarLongWritable,VarLongWritable> output,
                  Reporter reporter) throws IOException {
    String[] tokens = ToEntityPrefsMapper.DELIMITER.split(value.toString());
    long userID = Long.parseLong(tokens[0]);
    long itemID = Long.parseLong(tokens[1]);
    if (itemKey ^ transpose) {
      // If using items as keys, and not transposing items and users, then users are items!
      // Or if not using items as keys (users are, as usual), but transposing items and users,
      // then users are items! Confused?
      long temp = userID;
      userID = itemID;
      itemID = temp;
    }
    if (booleanData) {
      output.collect(new VarLongWritable(userID), new VarLongWritable(itemID));
    } else {
      float prefValue = tokens.length > 2 ? Float.parseFloat(tokens[2]) : 1.0f;
      output.collect(new VarLongWritable(userID), new EntityPrefWritable(itemID, prefValue));
    }
  }

}
